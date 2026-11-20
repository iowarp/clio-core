#ifndef CLIO_RUN_COMMANDS_H_
#define CLIO_RUN_COMMANDS_H_

/**
 * Command function declarations for the unified clio-run CLI
 * Each command function takes argc/argv with the subcommand args already stripped
 */

int RuntimeStart(int argc, char** argv);
int RuntimeStop(int argc, char** argv);
/** `pool start|stop|rm <pool> [--force]`: manage one pool (not the runtime). */
int Pool(int argc, char** argv);
int RuntimeStatus(int argc, char** argv);
int Monitor(int argc, char** argv);
int Compose(int argc, char** argv);
int Config(int argc, char** argv);
int Migrate(int argc, char** argv);
int RefreshRepo(int argc, char** argv);
int Viz(int argc, char** argv);

#endif  // CLIO_RUN_COMMANDS_H_
