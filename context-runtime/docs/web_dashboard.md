# The web dashboard ("viz")

The runtime serves its own dashboard. The admin ChiMod starts one HTTP server per
node, every ChiMod may ship a `viz/` directory of HTML/CSS/JS, and any Container
may register HTTP routes. Nothing about it is collective: each node answers from
its own state, or forwards an explicitly-addressed query, so N nodes serve the
same UI without coordinating.

This replaces the retired Python `context-visualizer` process entirely — the
dashboard, its module pages, and the management actions all live in the
runtime now.

## Running it

```bash
clio_run runtime start                      # dashboard on http://127.0.0.1:8080
clio_run runtime start --viz-port 9000      # elsewhere
clio_run runtime start --viz-bind 0.0.0.0   # reachable off-box (see below)
clio_run runtime start --no-viz             # don't serve it
```

Or in `~/.clio/clio.yaml`, which overrides the CLI's default:

```yaml
viz:
  enabled: true
  port: 8080
  bind: "127.0.0.1"
  max_threads: 16
```

Or in the environment, which overrides the file: `CLIO_VIZ_ENABLE`,
`CLIO_VIZ_PORT` (0 = bind any free port), `CLIO_VIZ_BIND`,
`CLIO_VIZ_MAX_THREADS`, `CLIO_VIZ_PATH`.

Two defaults are deliberate:

- **Loopback only.** The dashboard exposes worker queues, pool layout and device
  inventory. Binding `0.0.0.0` publishes that to the network; prefer an SSH
  tunnel. There is no authentication.
- **Off for embedded runtimes.** A unit test, an adapter, or any process calling
  `CLIO_INIT` gets no listening socket unless it asks. `clio_run runtime start`
  is what turns the dashboard on by default, so a daemon has it and a library
  user does not.

A port that is already taken logs a warning and disables the dashboard; it never
fails the runtime.

## Where the pages come from

A ChiMod's assets are found **relative to the library the runtime actually
loaded**, which is what makes a built-but-not-installed tree and an installed
tree both work with no configuration:

| layout | libraries | assets |
| --- | --- | --- |
| built, not installed | `<build>/bin` | `<build>/bin/viz/<mod_name>` |
| installed | `<prefix>/lib` | `<prefix>/share/clio/viz/<mod_name>` |

`$CLIO_VIZ_PATH` (a `:`-separated list of viz roots, each holding one
subdirectory per ChiMod) is checked first, so you can edit pages against a
running daemon without rebuilding.

CMake puts them in both places from one call in the module's `CMakeLists.txt`:

```cmake
clio_add_viz_assets(MOD_NAME clio_bdev)   # defaults to ./viz
```

Assets mount at `/viz/<mod_name>/`, and `/` redirects to whichever module claimed
the home page (the admin ChiMod's dashboard shell).

## Adding a page to your ChiMod

1. Drop the files in `viz/` next to your module's sources and call
   `clio_add_viz_assets(MOD_NAME <your_mod_name>)`. Your page is now at
   `/viz/<your_mod_name>/index.html`; the admin shell's stylesheet and helpers
   are served from the same origin, so you can `<link>`/`<script>` them:

   ```html
   <link rel="stylesheet" href="/viz/clio_admin/css/style.css">
   <script src="/viz/clio_admin/js/api.js"></script>
   ```

2. If the data your page needs is already a `Monitor()` query your module
   answers, you need no C++ at all — fetch it through the generic forward:

   ```js
   const data = await API.get(`/api/pools/${poolId}/monitor?query=stats`);
   ```

3. For anything else, override `Container::RegisterViz`:

   ```cpp
   void Runtime::RegisterViz(clio::run::viz::VizServer &viz,
                             const std::string &mod_name) {
     viz.AddRoute({"GET", "/api/mod/" + mod_name + "/pools", mod_name,
                   "Block-device pools on this node",
                   [](const clio::run::viz::Request &req,
                      clio::run::viz::Response &resp) {
                     clio::run::viz::JsonWriter w;
                     w.BeginObject();
                     w.Field("hello", req.Param("who", "world"));
                     w.EndObject();
                     resp.Json(w.Str());
                   }});
   }
   ```

   `RegisterViz` is called at module-LOAD time on a throwaway
   default-constructed prototype instance (so your create form and pages exist
   before any pool of the module does) and again per container by
   `PoolManager::RegisterContainer`. Registration is idempotent (first
   `(method, path)` wins). Because of the prototype call, neither the method
   body nor any handler may read container state or capture `this` — capture by
   value and construct clients locally (`Client(pool_id)` resolved from the
   request), as above.

   Path segments of the form `{name}` bind into `Request::path_vars`.

   `clio_runtime/viz/viz_json.h` is a dependency-free JSON writer, so a module
   does not inherit the dashboard's HTTP dependencies to answer a route.

`GET /api/routes` lists every registered route and mounted directory, which is
how the Config page discovers which modules shipped a UI.

## What a handler may do

Handlers run on the dashboard's own HTTP thread pool, **not** on a runtime
worker. That means:

- **Blocking is allowed.** Reading manager state (`CLIO_POOL_MANAGER`,
  `CLIO_IPC`, `CLIO_WORK_ORCHESTRATOR`, `CLIO_CONFIG_MANAGER`) is fine, and so is
  submitting a task and waiting on its `Future` — the same thing a FUSE thread or
  a client process does. Always pass a timeout; the admin's own forwards use 5s
  so an unreachable peer cannot pin an HTTP thread.
- **A handler is not a coroutine.** There is no `co_await` here.
- **Handlers must be reentrant.** Several may run at once.

The server is stopped early in `RuntimeManager::ServerFinalize`, before the
workers stop, so no handler can be left waiting on a task that will never run.

## Endpoints the admin ChiMod registers

| route | what it answers |
| --- | --- |
| `GET /api/health` | liveness plus this node's identity |
| `GET /api/topology` | cluster membership and SWIM state, from the local host table |
| `GET /api/pools` | pools composed on this node |
| `GET /api/config` | the settings this daemon actually came up with |
| `GET /api/routes` | every route and asset mount, per ChiMod |
| `GET /api/nodes/{node}/workers` | per-worker queue depth, load, task counts |
| `GET /api/nodes/{node}/system_stats` | CPU/RAM/GPU samples newer than `?min_event_id` |
| `GET /api/nodes/{node}/containers` | containers and their learned task-cost models |
| `GET /api/nodes/{node}/bdevs` | block devices on the node |
| `GET /api/pools/{pool}/monitor` | forward `?query` to a pool's own `Monitor()` (`?routing=local\|broadcast\|…`) |
| `GET /api/chimods` | loaded modules, and which registered a create form / shipped pages |
| `POST /api/pools/compose` | create a pool of ANY module via the compose path (identity fields + raw YAML) |
| `POST /api/pools/{pool}/destroy` | shut a pool down (the Pools tab's per-card "x"; the admin pool is refused) |

`{node}` is a node id or `local`. Addressing this node by its own id routes
locally, so the single-node case never touches the network.

The GETs are all read-only. The POSTs create pools and manage module resources
— the same operations any local RPC client can already perform — and stay
node-local; nothing here shuts a node down or deletes data.

## Creating pools: the "Add Pool" convention

The Pools tab's **Add Pool** button lists every loaded ChiMod
(`GET /api/chimods`) with a search box. Creation then goes one of two ways:

- **The module's own form.** A ChiMod registers
  `GET /api/mod/<mod>/create` answering a form spec —
  `{"title", "note", "fields":[{name,label,type,required,default,options,placeholder,help}]}`
  with field types `text`, `select`, `textarea` — and
  `POST /api/mod/<mod>/create` accepting those fields urlencoded.
  `action=validate` checks everything and creates nothing (the form's Validate
  button); otherwise the module creates the pool with its own typed client and
  answers `{"ok":true,"pool_id":...}`. Validation failures come back as
  `{"ok":false,"errors":{field: message}}` at HTTP 400, all fields at once.
  bdev, safe-bdev and the CTE core all register this convention.

- **The generic compose editor.** Modules without a form are still creatable:
  `POST /api/pools/compose` takes `mod_name`, `pool_name`, `pool_id`,
  `pool_query` and a raw-YAML `config` field, and drives the same
  `AsyncCompose` path as `clio_run compose` — the YAML is the compose entry's
  module parameters, verbatim.

Two rules make the forms work:

- **Explicit pool ids.** The runtime rejects null pool ids, so specs prefill
  `pool_id` from `viz::SuggestFreePoolMajor(base)` — a node-local suggestion
  the user may edit, not a reservation.
- **Never `ConfigParse::ParseSize` on form input.** It `exit(1)`s on garbage.
  Use `viz::ParseSizeField`, which rejects instead of dying.

POST bodies of type `application/x-www-form-urlencoded` are parsed into
`Request::params` (query string wins on a key collision), so handlers read one
map whether the value came from the URL or the form.

## The Pools tab

Pools are grouped into one section per ChiMod. Each pool is a card: clicking it
opens the pool's website — its module's page with `?pool=<id>` preselected, or
the generic pool page (`/viz/clio_admin/pool.html`) for modules that ship none —
and the card's corner "x" shuts the pool down after a confirmation
(`POST /api/pools/{pool}/destroy`; destroying the admin pool is refused, since
that is the runtime itself). Every pool page shows the pool's **task
predictions**: the learned per-method CPU/wall coefficients and their MAPE
(average prediction error), served by `GET /api/nodes/{node}/containers` and
rendered by the shared `renderPredictions()` helper in
`/viz/clio_admin/js/api.js`.

Known issue #1000: destroying a pool whose module runs periodic tasks
(safe-bdev, CTE) currently leaves those periodics retrying against the
destroyed pool.

## The module websites

- **`/viz/clio_bdev`** — per-device capacity, throughput, latency and TTL
  cards, from each pool's `Monitor("stats")`.
- **`/viz/clio_safe_bdev`** — pick an array, watch recovery progress and the
  member roster live, **add** a member (an existing bdev pool by name, or
  name+capacity to create one on the spot; `as_parity=1` raises the parity
  level) and **remove** one. Backed by
  `POST /api/mod/clio_safe_bdev/{pool}/add_member` and `.../remove_member`.
- **`/viz/clio_cte_core`** — the pool's storage-target roster (name, score,
  free space, traffic) with **register** (a bdev by name — fresh
  type+capacity, or `attach_pool_id` to attach an existing pool such as a
  safe-bdev array) and **unregister** buttons. Backed by
  `POST /api/mod/clio_cte_core/{pool}/register_target` and
  `.../unregister_target`, with the roster at `GET .../targets`.

## Build requirements

The HTTP transport is `Poco::Net`, linked PRIVATE into `clio_run_cxx` (Poco's
interface compile definitions collide with the POSIX interception adapters, so no
Poco type may appear in an installed header — the server lives behind a PIMPL in
`src/viz/viz_server.cc`). Without Poco the router still builds and the runtime
still starts; `VizServer::Start()` warns and the dashboard is simply absent
(`CLIO_RUN_HAS_POCO` is 0).

Tests: `ctest -R cr_viz_tests` covers the router directly (path patterns, MIME
types, JSON emission, asset resolution, the traversal guard, both asset layouts)
and then drives the real server over TCP, including a second ChiMod's page and
route appearing when its pool is composed.
