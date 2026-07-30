import java.util.ArrayList;
import java.util.List;

/**
 * One-JVM load+run driver for the CLIO_WITH_RUNTIME=1 (co-located) YCSB
 * experiment: the embedded runtime's store dies with the process, so the load
 * and transaction phases must share a JVM. YCSB's site.ycsb.Client.main ends
 * each phase with System.exit; a SecurityManager exit trap (JDK 21 still
 * honors it under -Djava.security.manager=allow) converts that into an
 * exception so the second phase can run.
 *
 * Usage: ClioBench <workload-file> <records> <ops> <threads> <export-prefix>
 */
public final class ClioBench {
  private static void mark(String m) { System.err.println(m); System.err.flush(); }

  private static final class ExitTrapped extends SecurityException {
    final int status;
    ExitTrapped(int status) { this.status = status; }
  }

  @SuppressWarnings("removal")
  public static void main(String[] args) throws Exception {
    String workload = args[0];
    String records = args[1];
    String ops = args[2];
    String threads = args[3];

    System.setSecurityManager(new SecurityManager() {
      @Override public void checkExit(int status) { throw new ExitTrapped(status); }
      @Override public void checkPermission(java.security.Permission p) {}
      @Override public void checkPermission(java.security.Permission p, Object c) {}
    });

    List<String> base = new ArrayList<>(List.of(
        "-db", "site.ycsb.db.ClioClient", "-P", workload,
        "-p", "recordcount=" + records, "-p", "operationcount=" + ops,
        "-threads", threads));

    mark("==== PHASE load ====");
    List<String> load = new ArrayList<>(base);
    load.add(0, "-load");
    load.add("-p"); load.add("exportfile=" + args[4] + ".load");
    try {
      site.ycsb.Client.main(load.toArray(new String[0]));
    } catch (ExitTrapped e) {
      mark("LOAD EXIT status=" + e.status);
      if (e.status != 0) { return; }
    } catch (Throwable t) {
      mark("LOAD THREW: " + t);
      t.printStackTrace(System.err);
      return;
    }

    mark("==== PHASE run ====");
    List<String> run = new ArrayList<>(base);
    run.add(0, "-t");
    run.add("-p"); run.add("exportfile=" + args[4] + ".run");
    try {
      site.ycsb.Client.main(run.toArray(new String[0]));
      mark("RUN RETURNED NORMALLY");
    } catch (ExitTrapped e) {
      mark("RUN EXIT status=" + e.status);
    } catch (Throwable t) {
      mark("RUN THREW: " + t);
      t.printStackTrace(System.err);
    }
    mark("==== PHASES DONE ====");
  }
}
