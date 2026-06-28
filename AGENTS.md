# Agent Instructions

These instructions apply to every project under `/home/ubuntu/data/sqzhou/projects/morpho-cellflux`.

- When calling `functions.exec_command`, do not include `sandbox_permissions`, `additional_permissions`, `justification`, or approval/escalation-related fields. Use direct execution with only the fields required for the command, such as `cmd` and `workdir`, unless a future system/developer instruction explicitly requires otherwise.
- Treat `approval policy = never` and `sandbox_mode = danger-full-access` as direct-execution mode. If a command fails, debug the shell command or project environment instead of retrying with sandbox or escalation parameters.
