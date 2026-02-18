#ifndef CHIMAERA_COMMANDS_H_
#define CHIMAERA_COMMANDS_H_

/**
 * Command function declarations for the unified chimaera CLI
 * Each command function takes argc/argv with the subcommand args already stripped
 */

int RuntimeStart(int argc, char** argv);
int RuntimeStop(int argc, char** argv);
int Monitor(int argc, char** argv);
int Compose(int argc, char** argv);
int Migrate(int argc, char** argv);
int RefreshRepo(int argc, char** argv);

#endif  // CHIMAERA_COMMANDS_H_
