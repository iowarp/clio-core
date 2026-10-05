#include <vector>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

// Magnetic reconnection in a Harris equilibrium thin current sheet
//
// This input deck reproduces the PIC simulations found in:
//   William Daughton. "Nonlinear dynamics of thin current sheets." Phys.
//   Plasmas. 9(9): 3668-3678. September 2002.
//
// This input deck was written by:
//   Kevin J Bowers, Ph.D.
//   Plasma Physics Group (X-1)
//   Applied Physics Division
//   Los Alamos National Lab
// August 2003      - original version
// October 2003     - heavily revised to utilize input deck syntactic sugar
// March/April 2004 - rewritten for domain decomposition V4PIC
 
// If you want to use global variables (for example, to store the dump
// intervals for your diagnostics section), it must be done in the globals
// section. Variables declared the globals section will be preserved across
// restart dumps. For example, if the globals section is:
//   

//////////////////////////////////////////////////////
// VPIC Weibel instability -- the Clio/NeuroPress compression workload.
//
// Derived from vpic-kokkos sample/Weibel/Weibel.cxx. Two changes, and nothing
// else about the physics is touched:
//
//   1. The grid is 3D and env-configurable (VPIC_NX/NY/NZ, VPIC_NPPC,
//      VPIC_STEPS, VPIC_DUMP_INT). Upstream's deck is 64 x 1 x 1, which is
//      4 KB of field data -- fine as a physics regression, useless as a
//      compression workload.
//
//   2. begin_diagnostics dumps the field array as flat float32 files, one per
//      field variable, when VPIC_DUMP_FIELDS=1. That is the same shape the Nyx
//      benchmark uses, so one replay driver reads both.
//
// Nothing here links or calls a compressor. The files it writes are equally
// usable by Clio or by upstream NeuroPress, which is the point: the workload
// and the thing being measured stay separate.
//
// The Weibel instability is a good compression subject for the same reason the
// Sedov blast is: it starts from a smooth counter-streaming equilibrium (highly
// compressible) and grows magnetic filaments that progressively fill the
// domain, so a single run spans a wide range of chunk compressibility.
//////////////////////////////////////////////////////

//     double variable;
//   } end_globals
// the double "variable" will be visible to other input deck sections as
// "global->variable". Note: Variables declared in the globals section are set
// to zero before the user's initialization block is executed. Up to 16K
// of global variables can be defined.
 
begin_globals {
  int    clio_dump_interval;   // steps between raw field dumps (0 = off)
  char   clio_dump_dir[512];
  char   clio_dump_vars[256];  // VPIC_DUMP_VARS: field names to dump ("" = all 16)
  double energies_interval;
  double fields_interval;
  double ehydro_interval;
  double ihydro_interval;
  double eparticle_interval;
  double iparticle_interval;
  double restart_interval;
};
 
begin_initialization {
  // At this point, there is an empty grid and the random number generator is
  // seeded with the rank. The grid, materials, species need to be defined.
  // Then the initial non-zero fields need to be loaded at time level 0 and the
  // particles (position and momentum both) need to be loaded at time level 0.
 
  // Arguments can be passed from the command line to the input deck
  // if( num_cmdline_arguments!=3 ) {
  //   sim_log( "Usage: " << cmdline_argument[0] << " mass_ratio seed" );
  //   abort(0);
  // }
  seed_entropy(1); //seed_entropy( atoi( cmdline_argument[2] ) );
 
  // Diagnostic messages can be passed written (usually to stderr)
  sim_log( "Computing simulation parameters");
 
  // Define the system of units for this problem (natural units)
  //double L    = 1; // Length normalization (sheet thickness)
  double de   = 1; // Length normalization (electron inertial length)
  double ec   = 1; // Charge normalization
  double me   = 1; // Mass normalization
  double c    = 1; // Speed of light
  double eps0 = 1; // Permittivity of space
 
  // Physics parameters
  double mi_me   = 1836; //25; //atof(cmdline_argument[1]); // Ion mass / electron mass
  // THE TEMPERATURE ANISOTROPY IS THE INSTABILITY'S FREE ENERGY, so it is the
  // one physics knob this deck exposes. Upstream's Weibel.cxx hardcodes
  // vth_perp = 0.25/sqrt(2) and vth_x = 0.05/sqrt(2), an anisotropy of
  // (vthe/vthex)^2 = 25; those remain the defaults, so an unset environment
  // reproduces upstream exactly. VPIC_VTHE and VPIC_VTHEX override them.
  //
  // Weibel grows because the distribution is hotter across x than along it:
  // the linear growth rate rises with both the perpendicular thermal velocity
  // and the anisotropy, so raising VPIC_VTHE or lowering VPIC_VTHEX makes the
  // magnetic filaments emerge from the noise sooner and the field arrays
  // change faster per timestep. See ../vpic/README.md and the growth-rate
  // measurements there.
  //
  // Ions track the electrons, as upstream sets them: vthi = vthe and
  // vthix = vthex. At mi/me = 1836 they are effectively immobile and the
  // instability is electron driven either way; keeping the coupling means the
  // deck stays a two-line change from upstream's.
  double vthe = getenv("VPIC_VTHE")  ? atof(getenv("VPIC_VTHE"))  : 0.25/sqrt(2.0);
  double vthex = getenv("VPIC_VTHEX") ? atof(getenv("VPIC_VTHEX")) : 0.05/sqrt(2.0);
  double vthi = vthe;   // Ion thermal velocity
  double vthix = vthex; // Ion thermal velocity in x-direction.
 
  double n0      = 1.0;    //  Background plasma density
  double b0 = 0.0;         // In plane magnetic field.
  double bg = 0.0;         // Guide field magnitude                         
  double tauwpe    = 200000;    // simulation wpe's to run

  // Numerical parameters
  double topology_x = nproc();  // Number of domains in x, y, and z
  double topology_y = 1; 
  double topology_z = 1;  // For load balance, best to keep "1" or "2" for Harris sheet
  // Cubic box: the 1x1 slab upstream uses makes sense for a 1D run, not for a
  // 3D one where the filaments should be isotropic.
  double Lx        = 10;
  double Ly        = 10;
  double Lz        = 10;
  // VPIC_L: box length (same on every axis). With the frozen vacuum fields of
  // LAYERED SLABS, use L = cells * 2^-5 so the cell size is an exact power of
  // two: the start-up divergence clean then sees an exactly zero error in the
  // vacuum slabs and leaves their fields bit-exact (with L = 10 it adds
  // rounding-level noise to every cell).
  if( getenv("VPIC_L") ) Lx = Ly = Lz = atof( getenv("VPIC_L") );
  // 3D and configurable. Defaults give 128^3 voxels -> 16 field vars x
  // 2,097,152 cells x 4 B = 134 MB per dump, which is the same order as the
  // Nyx benchmark's per-frame payload and lets the two be compared.
  // nppc is the memory driver: 128^3 x 8 = 16.8M macro-particles per species.
  double nx        = getenv("VPIC_NX")   ? atof(getenv("VPIC_NX"))   : 128;
  double ny        = getenv("VPIC_NY")   ? atof(getenv("VPIC_NY"))   : 128;
  double nz        = getenv("VPIC_NZ")   ? atof(getenv("VPIC_NZ"))   : 128;
  double nppc      = getenv("VPIC_NPPC") ? atof(getenv("VPIC_NPPC")) : 8;
  // VPIC_CELL: cell size on every axis (box length = cells * VPIC_CELL), for
  // a non-cubic box with cubic cells; overrides VPIC_L. With LAYERED SLABS use
  // a power of two (2^-5) for the same reason as VPIC_L.
  if( getenv("VPIC_CELL") ) {
    const double cell = atof( getenv("VPIC_CELL") );
    Lx = nx * cell; Ly = ny * cell; Lz = nz * cell;
  }
  double cfl_req   = 0.99f; //0.99;  // How close to Courant should we try to run
  double wpedt_max = 0.36;  // How big a timestep is allowed if Courant is not too restrictive
  double damp      = 0.0; // Level of radiation damping

 
  // Derived quantities
  double mi = me*mi_me;             // Ion mass
  double wpe  = c/de;               // electron plasma frequency
  double wpi  = wpe/sqrt(mi_me);    // ion plasma frequency
  double di   = c/wpi;              // ion inertial length
 
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;

  // LAYERED SLABS (VPIC_SLABS, off by default). The box is cut along z into
  // slabs of VPIC_SLAB_CELLS cells (16 by default), one character per slab.
  // Slab s holds cells k = s*SLAB .. s*SLAB+SLAB-1 of the 1-based cell index
  // (slab 0 starts at k = 1): the dumped array includes one ghost plane, so
  // array plane k is cell k and, at 254^3 with 16 z-planes per 4 MiB chunk,
  // every chunk of a field lies in exactly one slab.
  //   P  plasma: the bi-Maxwellian electrons and ions, at density n0
  //   C  vacuum holding a frozen "clumpy" electrostatic field E = -grad(phi),
  //      phi an integer level 0..VPIC_PHI_LEVELS-1 drawn per node, times
  //      VPIC_E_UNIT (a power of two): E takes a handful of exact values
  //   S  vacuum holding a frozen smooth field E = -grad(phi) + E0 z^, phi a
  //      product of sines (peak field VPIC_E_SMOOTH) rounded to a 2^-22 grid
  //      so every difference is exact, and E0 = VPIC_E0 a uniform offset in
  //      ez only (normal to the slabs, so it adds no curl)
  // A curl-free E in vacuum with B = 0 and no current does not change (the
  // clumpy one exactly: its differences are exact), until plasma streaming
  // out of the P slabs reaches it. Slabs past the pattern's end are plasma.
  const std::string slabs = getenv("VPIC_SLABS") ? getenv("VPIC_SLABS") : "";
  const int slab_cells = getenv("VPIC_SLAB_CELLS") ? atoi(getenv("VPIC_SLAB_CELLS")) : 16;
  std::vector<int> pslab;
  // Cells [lo, hi] (1-based) of slab s.
  auto slab_lo = [&](int s0) { return std::max(1, s0 * slab_cells); };
  auto slab_hi = [&](int s0) { return std::min((int)nz, s0 * slab_cells + slab_cells - 1); };
  for (int s0 = 0; s0 * slab_cells <= (int)nz; ++s0)
    if (s0 >= (int)slabs.size() || slabs[s0] == 'P') pslab.push_back(s0);
  double plasma_frac = 1.0;
  if (!slabs.empty()) {
    double pc = 0;
    for (int s0 : pslab) pc += std::max(0, slab_hi(s0) - slab_lo(s0) + 1);
    plasma_frac = pc / nz;
  }

  double Npe = n0*Ly*Lz*Lx*plasma_frac;    // Number physical electrons.
  double Npi = Npe;            // Number of physical ions in box
  double Ne  = nppc*nx*ny*nz*plasma_frac;  // total macro electrons in box

  Ne = trunc_granular(Ne,nproc());
  double Ni   = Ne;                                   // Total macro ions in box
  double qe = -ec*Npe/Ne;  // Charge per macro electron
  double qi =  ec*Npe/Ne;  // Charge per macro electron       
 
  double we   = Npe/Ne;                               // Weight of a macro electron
  double wi   = Npi/Ni;                               // Weight of a macro ion
 

  // Determine the timestep
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);      // Courant length
  double dt = cfl_req*dg/c;                           // Courant limited time step
  // printf("in harris.cxx: dt=%.7f\n",  dt);
  // exit(1);
  if( wpe*dt>wpedt_max ) dt=wpedt_max/wpe;            // Override time step if plasma frequency limited
  // VPIC_DT_DX: dt = VPIC_DT_DX * cell size / c (must stay under the Courant
  // limit, 1/sqrt(3) for cubic cells). With LAYERED SLABS use 0.5 and a
  // power-of-two cell size (VPIC_L): the B update's coefficients are then
  // powers of two, so curl(E) of the frozen clumpy field is computed exactly
  // as zero even with fused multiply-adds, and B stays exactly zero there.
  if( getenv("VPIC_DT_DX") ) dt = atof( getenv("VPIC_DT_DX") ) * hx / c;
 
  ////////////////////////////////////////
  // Setup high level simulation parmeters
 
  num_step             = getenv("VPIC_STEPS") ? atoi(getenv("VPIC_STEPS")) : 200;
  status_interval      = 0; //2000;
  sync_shared_interval = 0; //status_interval;

  // DIVERGENCE CLEANING, and why it is not merely a physics preference here.
  //
  // Upstream's deck disables it ("turn off cleaning (GY)"). The consequence for
  // a COMPRESSION benchmark is that div_e_err, div_b_err, rhob and rhof are
  // never recomputed: they hold their initial values for the whole run and are
  // dumped unchanged every frame. Measured at 126^3/200 steps, 8 frames: those
  // four are BIT-IDENTICAL across every consecutive pair, they are 25% of the
  // payload, and because they compress ~4.2x against the 1.06x the twelve
  // evolving fields manage, they inflate the run's headline ratio from 1.061
  // to 1.304 -- a 1.23x lift from data the simulation never touched.
  //
  // Non-zero makes them real diagnostics that evolve with the fields. It costs
  // a Poisson-ish solve every N steps, so it is opt-in rather than defaulted
  // on, and the default stays upstream's 0 so existing numbers are unchanged.
  clean_div_e_interval = getenv("VPIC_CLEAN_DIV_INT")
                             ? atoi(getenv("VPIC_CLEAN_DIV_INT")) : 0;
  clean_div_b_interval = clean_div_e_interval;
 
  // Raw field dump for the compression benchmark. Off unless
 
  // VPIC_DUMP_FIELDS=1; interval in steps, directory from VPIC_DUMP_DIR.
 
  global->clio_dump_interval =
 
      (getenv("VPIC_DUMP_FIELDS") && atoi(getenv("VPIC_DUMP_FIELDS")))
 
          ? (getenv("VPIC_DUMP_INT") ? atoi(getenv("VPIC_DUMP_INT")) : 25)
 
          : 0;
 
  {
 
    const char* d = getenv("VPIC_DUMP_DIR");
 
    snprintf(global->clio_dump_dir, sizeof(global->clio_dump_dir), "%s",
 
             d ? d : ".");
 
    const char* dv = getenv("VPIC_DUMP_VARS");
    snprintf(global->clio_dump_vars, sizeof(global->clio_dump_vars), "%s",
             dv ? dv : "");
  }


 
  global->energies_interval  = 1; //000; //status_interval;
  global->fields_interval    = status_interval;
  global->ehydro_interval    = status_interval;
  global->ihydro_interval    = status_interval;
  global->eparticle_interval = status_interval; // Do not dump
  global->iparticle_interval = status_interval; // Do not dump
  global->restart_interval   = status_interval; // Do not dump
 
  ///////////////////////////
  // Setup the space and time
 
  // Setup basic grid parameters
  define_units( c, eps0 );
  define_timestep( dt );
  grid->dx = hx;
  grid->dy = hy;
  grid->dz = hz;
  grid->dt = dt;
  grid->cvac = c;
  //grid->damp = damp;

  // Parition a periodic box among the processors sliced uniformly along y
  // define_periodic_grid( -0.5*Lx, 0, 0,    // Low corner
  //                        0.5*Lx, Ly, Lz,  // High corner
  //                        nx, ny, nz,      // Resolution
  //                        1, nproc(), 1 ); // Topology
  define_periodic_grid(  0, -0.5*Ly, -0.5*Lz,    // Low corner
			  Lx, 0.5*Ly, 0.5*Lz,     // High corner
			  nx, ny, nz,             // Resolution
			  topology_x, topology_y, topology_z); // Topology

  //   printf("in harris.cxx: g->neighbor[6*265]=%jd\n",  grid->neighbor[6*265]);
  // Override some of the boundary conditions to put a particle reflecting
  // perfect electrical conductor on the -x and +x boundaries
  // set_domain_field_bc( BOUNDARY(-1,0,0), pec_fields );
  // set_domain_field_bc( BOUNDARY( 1,0,0), pec_fields );
  // set_domain_particle_bc( BOUNDARY(-1,0,0), reflect_particles );
  // set_domain_particle_bc( BOUNDARY( 1,0,0), reflect_particles );
 
  define_material( "vacuum", 1 );
  // Note: define_material defaults to isotropic materials with mu=1,sigma=0
  // Tensor electronic, magnetic and conductive materials are supported
  // though. See "shapes" for how to define them and assign them to regions.
  // Also, space is initially filled with the first material defined.
 
  // If you pass NULL to define field array, the standard field array will
  // be used (if damp is not provided, no radiation damping will be used).
  define_field_array( NULL, damp );
 
  ////////////////////
  // Setup the species
 
  // Allow 50% more local_particles in case of non-uniformity
  // VPIC will pick the number of movers to use for each species
  // Both species use out-of-place sorting
  // species_t * ion      = define_species( "ion",       ec, mi, 1.5*Ni/nproc(), -1, 40, 1 );
  // species_t * electron = define_species( "electron", -ec, me, 1.5*Ne/nproc(), -1, 20, 1 );
  //species_t *electron = define_species("electron",-ec,me,2.4*Ne/nproc(),-1,25,0);
  //species_t *ion      = define_species("ion",      ec,mi,2.4*Ne/nproc(),-1,25,0);

  species_t *electron = define_species("electron",-ec,me,2.4*Ne/nproc(),-1,0,0); //turn off sorting (GY)
  species_t *ion      = define_species("ion",      ec,mi,2.4*Ne/nproc(),-1,0,0); //(GY) 
 
  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation
 
  sim_log( "***********************************************" );
  sim_log ( "mi/me = " << mi_me );
  sim_log ( "tauwpe = " << tauwpe );
  sim_log ( "num_step = " << num_step );
  sim_log ( "Lx/di = " << Lx/di );
  sim_log ( "Lx/de = " << Lx/de );
  sim_log ( "Ly/di = " << Ly/di );
  sim_log ( "Ly/de = " << Ly/de );
  sim_log ( "Lz/di = " << Lz/di );
  sim_log ( "Lz/de = " << Lz/de );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz ); 
  sim_log ( "damp = " << damp );
  sim_log ( "courant = " << c*dt/dg );
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc = " << nppc );
  sim_log ( " b0 = " << b0 );
  sim_log ( " di = " << di );
  sim_log ( " Ne = " << Ne );
  sim_log ( "total # of particles = " << 2*Ne );
  sim_log ( "dt*wpe = " << wpe*dt ); 
  sim_log ( "dx/de = " << Lx/(de*nx) );
  sim_log ( "dy/de = " << Ly/(de*ny) );
  sim_log ( "dz/de = " << Lz/(de*nz) );
  sim_log ( "dx/debye = " << (Lx/nx)/(vthe/wpe)  );
  sim_log ( "n0 = " << n0 );
  sim_log ( "vthi/c = " << vthi/c );
  sim_log ( "vthe/c = " << vthe/c );
  sim_log( "" );
 
  ////////////////////////////
  // Load fields and particles
 
  // sim_log( "Loading fields" );
 
  // set_region_field( everywhere, 0, 0, 0,                    // Electric field
  //                   0, -sn*b0*tanh(x/L), cs*b0*tanh(x/L) ); // Magnetic field
  // Note: everywhere is a region that encompasses the entire simulation
  // In general, regions are specied as logical equations (i.e. x>0 && x+y<2)

  if( !slabs.empty() ) {
    // Frozen vacuum fields of the C and S slabs (see LAYERED SLABS above).
    const int levels = getenv("VPIC_PHI_LEVELS") ? atoi(getenv("VPIC_PHI_LEVELS")) : 8;
    const double e_unit = getenv("VPIC_E_UNIT") ? atof(getenv("VPIC_E_UNIT")) : 0.0078125;
    const double e_smooth = getenv("VPIC_E_SMOOTH") ? atof(getenv("VPIC_E_SMOOTH")) : 0.3;
    const double e0_s = getenv("VPIC_E0") ? atof(getenv("VPIC_E0")) : 1.0;
    // Grid of the smooth potential: 2^-VPIC_Q_BITS (22 by default). Every E
    // value of the S slab is then a multiple of it, exact in float32 while
    // |E| < 2^(24 - bits) (4 for 22 bits), so the curl stays exactly zero.
    const double q23 = std::ldexp( 1.0, -( getenv("VPIC_Q_BITS") ? atoi(getenv("VPIC_Q_BITS")) : 22 ) );
    const int ni = (int)nx, nj = (int)ny, nk = (int)nz;
    const double kw = 2.0 * M_PI / Lx;
    // Node potential; indices wrap periodically over 1..n.
    auto phi = [&]( int i, int j, int k ) -> double {
      i = (i - 1 + ni) % ni + 1; j = (j - 1 + nj) % nj + 1; k = (k - 1 + nk) % nk + 1;
      const int s0 = k / slab_cells;
      const char t = s0 < (int)slabs.size() ? slabs[s0] : 'P';
      if( t == 'C' ) {
        unsigned int h = (unsigned)i * 73856093u ^ (unsigned)j * 19349663u ^
                         (unsigned)k * 83492791u;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
        return (double)( h % (unsigned)levels ) * e_unit;
      }
      if( t == 'S' ) {
        const double x = (i - 1) * hx, y = (j - 1) * hy, z = (k - 1) * hz;
        // sin^2 taper, exactly zero on the slab's first plane and small near
        // its last: E across each boundary (a difference of a clumpy level and
        // this) stays exact in float32, so the boundary adds no curl, and the
        // last plane of a neighbouring C chunk (whose ez reaches into this
        // slab's first plane) keeps the clumpy range.
        const double tz = sin( M_PI * ( k - s0 * slab_cells ) / slab_cells );
        return std::nearbyint( e_smooth / ( kw * hx ) * tz * tz *
                               sin( kw * x ) * sin( kw * y + 0.7 ) * sin( kw * z + 1.3 ) / q23 ) * q23;
      }
      return 0.0;
    };
    // Every voxel, ghosts included (0 .. n+1), from the same periodic phi:
    // VPIC averages the two copies of each periodic boundary face at start-up,
    // and a ghost left at zero would break E = -grad(phi) on those faces.
    for( int k = 0; k <= nk + 1; ++k )
      for( int j = 0; j <= nj + 1; ++j )
        for( int i = 0; i <= ni + 1; ++i ) {
          const double p0 = phi( i, j, k );
          field( i, j, k ).ex = (float)( -( phi( i + 1, j, k ) - p0 ) );
          field( i, j, k ).ey = (float)( -( phi( i, j + 1, k ) - p0 ) );
          const int sk = ( ( k - 1 + nk ) % nk + 1 ) / slab_cells;
          const bool s_slab = sk < (int)slabs.size() && slabs[sk] == 'S';
          field( i, j, k ).ez = (float)( ( s_slab ? e0_s : 0.0 ) - ( phi( i, j, k + 1 ) - p0 ) );
        }
    sim_log( "Slabs " << slabs << " x " << slab_cells << " cells; plasma fraction "
             << plasma_frac );
  }

  sim_log( "Loading particles" );
 
  // Do a fast load of the particles
  //seed_rand( rng_seed*nproc() + rank() );  //Generators desynchronized
  double xmin = grid->x0 , xmax = grid->x0+(grid->dx)*(grid->nx);
  double ymin = grid->y0 , ymax = grid->y0+(grid->dy)*(grid->ny);
  double zmin = grid->z0 , zmax = grid->z0+(grid->dz)*(grid->nz);
  
  sim_log( "-> Uniform Bi-Maxwellian" );
  int seed = 1;
  int seedn= 1;
  double n1,n2,n3;
  int signx,signy,signz;
  repeat ( Ne/nproc() ) {
  //repeat ( Ne/(8*nproc()) ) {
  //repeat ( 10 ) {

   double x = uniform( rng(0), xmin, xmax );
   double y = uniform( rng(0), ymin, ymax );
   double z;
   if( slabs.empty() ) {
     z = uniform( rng(0), zmin, zmax );
   } else {
     int q = (int)uniform( rng(0), 0, (double)pslab.size() );
     if( q >= (int)pslab.size() ) q = (int)pslab.size() - 1;
     const double zlo = zmin + ( slab_lo( pslab[q] ) - 1 ) * hz;
     const double zhi = zmin + slab_hi( pslab[q] ) * hz;
     z = uniform( rng(0), zlo, zhi );
   }
   n1 = normal(rng(0),0,vthex);
   n2 = normal(rng(0),0,vthe );
   n3 = normal(rng(0),0,vthe );

   inject_particle( electron, x, y, z,
		    n1,
		    n2,
		    n3,we, 0, 0);

   n1 = normal(rng(0),0,vthix);
   n2 = normal(rng(0),0,vthi );
   n3 = normal(rng(0),0,vthi );

   inject_particle( ion, x, y, z,
		    n1,
		    n2,
		    n3,wi, 0 ,0 );
   
  //   double x = uniform2(  xmin, xmax, seed );
  //   double y = uniform2(  ymin, ymax, seed );
  //   double z = uniform2(  zmin, zmax, seed );
  //   //printf("x=%.14f,y=%.14f,z=%.14f,seed=%d\n",x,y,z,seed);
  //   n1 = normal2( 0,vthex,seedn);
  //   n2 = normal2( 0,vthe ,seedn);
  //   n3 = normal2( 0,vthe ,seedn);
  //   //printf("n1=%.14f,n2=%.14f,n3=%.14f,seedn=%d\n",n1,n2,n3,seedn);
  //   signx = -1;
  //   signy = -1;
  //   signz = -1;
  //   for(int i=0; i<2; i++){
  //     signx = -signx;
  //     for(int j=0; j<2; j++){
  // 	signy = -signy;
  // 	for(int k=0; k<2; k++){
  // 	  signz = -signz;
  // inject_particle( electron, x, y, z,
  // 		   n1*signx,
  //                  n2*signy,
  //                  n3*signz,we, 0, 0);

  // 	}
  //     }
  //   }
  //   n1 = normal2( 0,vthix,seedn);
  //   n2 = normal2( 0,vthi ,seedn);
  //   n3 = normal2( 0,vthi ,seedn);
  //   //printf("n1=%.14f,n2=%.14f,n3=%.14f,seedn=%d\n",n1,n2,n3,seedn);

    
  //   signx = -1;
  //   signy = -1;
  //   signz = -1;
  //   for(int i=0; i<2; i++){
  //     signx = -signx;
  //     for(int j=0; j<2; j++){
  // 	signy = -signy;
  // 	for(int k=0; k<2; k++){
  // 	  signz = -signz;

  // inject_particle( ion, x, y, z,
  //                  n1*signx,
  //                  n2*signy,
  //                  n3*signz,wi, 0 ,0 );
  // 	}
  //     }
  //   }
  
  }

  sim_log( "Finished loading particles" );
 
  //exit(1);

  // Upon completion of the initialization, the following occurs:
  // - The synchronization error (tang E, norm B) is computed between domains
  //   and tang E / norm B are synchronized by averaging where discrepancies
  //   are encountered.
  // - The initial divergence error of the magnetic field is computed and
  //   one pass of cleaning is done (for good measure)
  // - The bound charge density necessary to give the simulation an initially
  //   clean divergence e is computed.
  // - The particle momentum is uncentered from u_0 to u_{-1/2}
  // - The user diagnostics are called on the initial state
  // - The physics loop is started
  //
  // The physics loop consists of:
  // - Advance particles from x_0,u_{-1/2} to x_1,u_{1/2}
  // - User particle injection at x_{1-age}, u_{1/2} (use inject_particles)
  // - User current injection (adjust field(x,y,z).jfx, jfy, jfz)
  // - Advance B from B_0 to B_{1/2}
  // - Advance E from E_0 to E_1
  // - User field injection to E_1 (adjust field(x,y,z).ex,ey,ez,cbx,cby,cbz)
  // - Advance B from B_{1/2} to B_1
  // - (periodically) Divergence clean electric field
  // - (periodically) Divergence clean magnetic field
  // - (periodically) Synchronize shared tang e and norm b
  // - Increment the time step
  // - Call user diagnostics
  // - (periodically) Print a status message
}
 
begin_diagnostics {
 
# define should_dump(x) (global->x##_interval>0 && remainder(step(),global->x##_interval)==0)

  /* ------------------------------------------------------------------
   * Raw field dump for the Clio/NeuroPress compression benchmark.
   *
   * VPIC keeps the field array as Kokkos::View<float*[FIELD_VAR_COUNT]>,
   * device-resident and INTERLEAVED: variable m of voxel v sits at
   * v*FIELD_VAR_COUNT + m. Writing that straight out would give one file whose
   * every 16th float belongs to the same physical quantity, which is not what
   * a compressor sees from any other writer and would make the numbers
   * incomparable with the Nyx benchmark's per-component files.
   *
   * So de-interleave: one flat float32 file per field variable, each a
   * contiguous n_voxels array. That is the same shape Nyx dumps, so one replay
   * driver reads both.
   *
   * Host mirror, not the device pointer: this deck deliberately links nothing
   * that could take a device pointer, so the copy is explicit and the files
   * are the only output.
   * ------------------------------------------------------------------ */
  /* step() > 0: at step 0 the field array is still identically zero -- the
     particles have been loaded but no field solve has run -- so a frame dumped
     there is 16 files of pure zeros. It compresses ~infinitely and would
     dominate any ratio it is averaged into, while describing no physics at
     all. Nyx does not have this problem (its step-0 state is the initial
     condition), which is why only this deck needs the guard. */
  if( global->clio_dump_interval > 0 && step() > 0 &&
      step() % global->clio_dump_interval == 0 ) {
    static const char* kFieldNames[16] = {
      "ex","ey","ez","div_e_err", "cbx","cby","cbz","div_b_err",
      "tcax","tcay","tcaz","rhob",  "jfx","jfy","jfz","rhof" };
    static int clio_frame = 0;

    Kokkos::deep_copy( field_array->k_f_h, field_array->k_f_d );
    const size_t nv = field_array->k_f_h.extent(0);

    char dir[640];
    snprintf( dir, sizeof(dir), "%s/plt%05d", global->clio_dump_dir, clio_frame );
    mkdir( global->clio_dump_dir, 0755 );
    mkdir( dir, 0755 );

    std::vector<float> col( nv );
    const std::string want = std::string(",") + global->clio_dump_vars + ",";
    for( int m = 0; m < FIELD_VAR_COUNT && m < 16; ++m ) {
      if( want.size() > 2 &&
          want.find( std::string(",") + kFieldNames[m] + "," ) == std::string::npos )
        continue;   // VPIC_DUMP_VARS lists the fields to write
      for( size_t v = 0; v < nv; ++v ) col[v] = field_array->k_f_h(v, m);
      char path[800];
      snprintf( path, sizeof(path), "%s/fab0000_comp%02d_%s.f32",
                dir, m, kFieldNames[m] );
      FILE* fp = fopen( path, "wb" );
      if( fp ) { fwrite( col.data(), sizeof(float), nv, fp ); fclose( fp ); }
    }
    if( rank() == 0 )
      sim_log( "[clio-dump] frame " << clio_frame << ": " << nv
               << " voxels x " << FIELD_VAR_COUNT << " vars" );
    ++clio_frame;
  }

 
  if( step()==-10 ) {
    // A grid dump contains all grid parameters, field boundary conditions,
    // particle boundary conditions and domain connectivity information. This
    // is stored in a binary format. Each rank makes a grid dump
    dump_grid("grid");
 
    // A materials dump contains all the materials parameters. This is in a
    // text format. Only rank 0 makes the materials dump
    dump_materials("materials");
 
    // A species dump contains the physics parameters of a species. This is in
    // a text format. Only rank 0 makes the species dump
    dump_species("species");
  }
 
  // Energy dumps store all the energies in various directions of E and B
  // and the total kinetic (not including rest mass) energies of each species
  // species in a simple text format. By default, the energies are appended to
  // the file. However, if a "0" is added to the dump_energies call, a new
  // energies dump file will be created. The energies are in the units of the
  // problem and are all time centered appropriately. Note: When restarting a
  // simulation from a restart dump made at a prior time step to the last
  // energies dump, the energies file will have a "hiccup" of intervening
  // time levels. This "hiccup" will not occur if the simulation is aborted
  // immediately following a restart dump. Energies dumps are in a text
  // format and the layout is documented at the top of the file. Only rank 0
  // makes makes an energies dump.
  if( should_dump(energies) ) {
    dump_energies( "energies", step()==0 ? 0 : 1 );
  }
  
  // Field dumps store the raw electromagnetic fields, sources and material
  // placement and a number of auxilliary fields. E, B and RHOB are
  // timecentered, JF and TCA are half a step old. Material fields are static
  // and the remaining fields (DIV E ERR, DIV B ERR and RHOF) are for
  // debugging purposes. By default, field dump filenames are tagged with
  // step(). However, if a "0" is added to the call, the filename will not be
  // tagged. The JF that gets stored is accumulated with a charge-conserving
  // algorithm. As a result, JF is not valid until at least one timestep has
  // been completed. Field dumps are in a binary format. Each rank makes a
  // field dump.
  if( step()==-10 )         dump_fields("fields"); // Get first valid total J
  if( should_dump(fields) ) dump_fields("fields");
 
  // Hydro dumps store particle charge density, current density and
  // stress-energy tensor. All these quantities are known at the time
  // t = time().  All these quantities are accumulated trilinear
  // node-centered. By default, species dump filenames are tagged with
  // step(). However, if a "0" is added to the call, the filename will not
  // be tagged. Note that the current density accumulated by this routine is
  // purely diagnostic. It is not used by the simulation and it is not
  // accumulated using a self-consistent charge-conserving method. Hydro dumps
  // are in a binary format. Each rank makes a hydro dump.
  if( should_dump(ehydro) ) dump_hydro("electron","ehydro");
  if( should_dump(ihydro) ) dump_hydro("ion",     "ihydro");
 
  // Particle dumps store the particle data for a given species. The data
  // written is known at the time t = time().  By default, particle dumps
  // are tagged with step(). However, if a "0" is added to the call, the
  // filename will not be tagged. Particle dumps are in a binary format.
  // Each rank makes a particle dump.
  if( should_dump(eparticle) ) dump_particles("electron","eparticle");
  if( should_dump(iparticle) ) dump_particles("ion",     "iparticle");
 
  // A checkpt is made by calling checkpt( fbase, tag ) where fname is a string
  // and tag is an integer.  A typical usage is:
  //   checkpt( "checkpt", step() ).
  // This will cause each process to write their simulation state to a file
  // whose name is based on fbase, tag and the node's rank.  For the above
  // usage, if called on step 314 on a 4 process run, the four files:
  //   checkpt.314.0, checkpt.314.1, checkpt.314.2, checkpt.314.3
  // to be written.  The simulation can then be restarted from this point by
  // invoking the application with "--restore checkpt.314".  checkpt must be 
  // the _VERY_ LAST_ diagnostic called.  If not, diagnostics performed after
  // the checkpt but before the next timestep will be missed on restore.
  // Restart dumps are in a binary format unique to the each simulation.

  if( should_dump(restart) ) checkpt( "checkpt", step() );

  // If you want to write a checkpt after a certain amount of simulation time,
  // use uptime() in conjunction with checkpt.  For example, this will cause
  // the simulation state to be written after 7.5 hours of running to the
  // same file every time (useful for dealing with quotas on big machines).
  //if( uptime()>=27000 ) {
  //  checkpt( "timeout", 0 );
  //  abort(0);
  //}
 
# undef should_dump
 
}
 
begin_particle_injection {
 
  // No particle injection for this simulation
 
}
 
begin_current_injection {
 
  // No current injection for this simulation
 
}
 
begin_field_injection {
 
  // No field injection for this simulation
 
}

begin_particle_collisions{ 

  // No collisions for this simulation 

}
