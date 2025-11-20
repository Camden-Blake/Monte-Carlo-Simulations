# import os
# from dataclasses import dataclass
# from pathlib import Path
# import re
# import subprocess

# @dataclass
# class Simulation:
#     location: Path
#     model_file: str
#     slurm_file: str
#     was_run: bool

#     def run_simulation(self) -> None:
#         if self.was_run:
#             print(f"[SKIP] Results already exist for {self.location}")
#             return

#         print(f"[RUN] Submitting job in {self.location} ...")
#         try:
#             subprocess.run(
#                 # ["echo", "sbatch", self.slurm_file],
#                 ["sbatch", self.slurm_file],
#                 cwd=self.location,
#                 check=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#             print(f"[OK] Submitted {self.slurm_file}")
#         except subprocess.CalledProcessError as e:
#             print(f"[ERROR] Failed to submit {self.slurm_file}")
#             print(e.stderr.decode().strip())

# def get_simulations(path: Path) -> list[Simulation]:
#     simulations = []
#     model_pattern = re.compile(r"^model\.*", re.IGNORECASE)
#     slurm_pattern = re.compile(r".*\.slurm$", re.IGNORECASE)
#     result_pattern = re.compile(r"^slurm-\d+\.(out|err)$", re.IGNORECASE)
#     for root, dirs, files in os.walk(path):
#         root_path = Path(root).absolute()
#         model_matches = [f for f in files if model_pattern.match(f)]
#         slurm_matches = [f for f in files if slurm_pattern.match(f)]
#         if model_matches and slurm_matches:
#             was_run = any(result_pattern.match(f) for f in files)
#             simulations.append(
#                 Simulation(
#                     location=root_path,
#                     model_file=model_matches[0],
#                     slurm_file=slurm_matches[0],
#                     was_run=was_run,
#                 )
#             )
#     return simulations


# if __name__ == "__main__":
#     code_paths = [
#         Path("MCNP"),
#         Path("OpenMC"),
#         ]
#     for code_path in code_paths:
#         simulations = get_simulations(code_path)
#         for sim in simulations:
#             sim.run_simulation()


from pathlib import Path
from Analysis.Modules.Simulations import get_simulations, CODE_ACCESSORS

if __name__ == "__main__":
    codes = ["MCNP", "OpenMC"]
    for code in codes:
        path = Path(code)
        accessor = CODE_ACCESSORS[code]
        simulations = get_simulations(path, accessor)
        for sim in simulations:
            sim.run_simulation()