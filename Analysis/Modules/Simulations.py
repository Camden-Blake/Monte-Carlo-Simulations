import re
import file_read_backwards as frb
import h5py as h5
import subprocess
import numpy as np
import os

import openmc

from dataclasses import dataclass, field
from typing import Dict, Generic, TypeVar
from pathlib import Path
from collections import defaultdict

### General Helpers
unit_to_seconds = {
    'second': 1,
    'seconds': 1,
    'minute': 60,
    'minutes': 60,
    'hour': 3600,
    'hours': 3600,
}

def get_directories(path: Path):
    return (item.stem for item in path.iterdir() if item.is_dir())

def get_result_file(path: Path, pattern: str):
    files = [item for item in path.glob(pattern) if item.is_file()]
    if len(files) != 1:
        raise RuntimeError(f"Expected one result file in {path}, found {len(files)}.")
    return files[0]

def nested_dict():
    return defaultdict(nested_dict)

def to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: to_dict(v) for k, v in d.items()}
    return d

def normalize_to_lethargy(data: Spectrum):
    for g in range(len(data.bounds)-1):
        emin = data.bounds[g]
        emax = data.bounds[g+1]
        u = np.log(emax/emin)
        data.values[g] /= u
        data.std_dev[g] /= u

### Types
T = TypeVar("T")

### Classes
@dataclass
class RunTime:
    """Holds the runtime."""
    val: float

@dataclass
class KResults:
    """Holds the k_eff and standard deviation from a k-code run."""
    val: float
    std: float

@dataclass
class Spectrum:
    """Holds the spectrum data from a tally."""
    bounds: np.ndarray
    values: np.ndarray
    std_dev: np.ndarray

@dataclass
class Profile:
    """Holds the profile data from a single tally."""
    e_grid: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    z_grid: np.ndarray
    values: np.ndarray
    std_dev: np.ndarray

@dataclass
class Profiles:
    profiles: Dict[str, Profile] = field(default_factory=dict)

@dataclass
class RunResult(Generic[T]):
    """Holds any type of result object for a given run."""
    name: str
    data: T

@dataclass
class ProblemResults(Generic[T]):
    """Holds multiple RunResults for a single problem."""
    name: str
    runs: Dict[str, RunResult[T]] = field(default_factory=dict)

    def add_run(self, run_name: str, data: T):
        self.runs[run_name] = RunResult(run_name, data)

@dataclass
class CodeResults(Generic[T]):
    """Holds all problem results for a given code."""
    name: str
    problems: Dict[str, ProblemResults[T]] = field(default_factory=dict)

    def add_result(self, problem: str, run: str, data: T):
        if problem not in self.problems:
            self.problems[problem] = ProblemResults(problem)
        self.problems[problem].add_run(run, data)

@dataclass
class SimulationResults(Generic[T]):
    """Top-level container for all codes."""
    codes: Dict[str, CodeResults[T]] = field(default_factory=dict)

    def add_result(self, code: str, problem: str, run: str, data: T):
        if code not in self.codes:
            self.codes[code] = CodeResults(code)
        self.codes[code].add_result(problem, run, data)

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

@dataclass
class Dimensions:
    x: int
    y: int
    z: int

class OpenMCMesh:
    lower_left: Coordinate
    upper_right: Coordinate
    dimension: Dimensions

    def __init__(self, mesh: openmc.MeshBase):
        self.lower_left = Coordinate(*mesh.bounding_box.lower_left)
        self.upper_right = Coordinate(*mesh.bounding_box.upper_right)
        self.dimension = Dimensions(*mesh.dimension)

    def return_x_grid(self):
        return self._return_grid(
            start = self.lower_left.x,
            end = self.upper_right.x,
            num = self.dimension.x + 1
        )

    def return_y_grid(self):
        return self._return_grid(
            start = self.lower_left.y,
            end = self.upper_right.y,
            num = self.dimension.y + 1
        )

    def return_z_grid(self):
        return self._return_grid(
            start = self.lower_left.z,
            end = self.upper_right.z,
            num = self.dimension.z + 1
        )

    def _return_grid(self, start:float, end:float, num:int) -> np.ndarray:
        return np.linspace(start=start, stop=end, num=num, endpoint=True)
    
@dataclass(frozen=True)
class CodeResultsAccessor:
    runtime_result_pattern: str
    runtime_normalization: float
    runtime_reader: callable

    k_eff_result_pattern: str
    k_eff_reader: callable

    spectrum_result_pattern: str
    spectrum_reader: callable

    profiles_result_pattern: str
    profiles_reader: callable

    def get_runtime(self, path: Path) -> float:
        return self.runtime_reader(get_result_file(path, self.runtime_result_pattern)) * self.runtime_normalization

    def get_k_eff(self, path: Path) -> KResults:
        return self.k_eff_reader(get_result_file(path, self.k_eff_result_pattern))

    def get_spectrum(self, path: Path, identifier: str) -> Spectrum:
        return self.spectrum_reader(get_result_file(path, self.spectrum_result_pattern), identifier)

    def get_profiles(self, path: Path, identifiers: list[str]) -> Profiles:
        return self.profiles_reader(get_result_file(path, self.profiles_result_pattern), identifiers)

@dataclass
class Simulation:
    location: Path
    problem: str
    run: str
    model_file: str
    slurm_file: str
    was_run: bool
    result_accessor: CodeResultsAccessor

    def run_simulation(self) -> None:
        if self.was_run:
            print(f"[SKIP] Results already exist for {self.location}")
            return

        print(f"[RUN] Submitting job in {self.location} ...")
        try:
            subprocess.run(
                # ["echo", "sbatch", self.slurm_file],
                ["sbatch", self.slurm_file],
                cwd=self.location,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"[OK] Submitted {self.slurm_file}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit {self.slurm_file}")
            print(e.stderr.decode().strip())
    

### OpenMC Functions
def read_openmc_runtime(path:Path) -> float:
    with h5.File(path, 'r') as file:
        return float(file["runtime/total"][()])
    raise RuntimeWarning(f"Unable to read runtime for {path}")

def read_openmc_k(path:Path) -> KResults:
    with h5.File(path, 'r') as file:
        keff = file["k_combined"][:]
        return KResults(val=keff[0], std=keff[1])
    raise RuntimeWarning(f"Unable to read keff for {path}")

def read_openmc_spectrum(path:Path, identifier:str) -> Spectrum:
    sp = openmc.StatePoint(path)
    tally = sp.get_tally(name=identifier)
    spectrum = Spectrum(
        bounds=np.array(tally.filters[0].values),
        values=np.array(tally.mean[:,0,0]),
        std_dev=np.array(tally.std_dev[:,0,0]),
    )
    normalize_to_lethargy(spectrum)
    return spectrum

def _read_openmc_profile(tally: openmc.Tally) -> Profile:
    energy_filter = tally.find_filter(openmc.EnergyFilter)
    mesh_info = OpenMCMesh(tally.find_filter(openmc.MeshFilter).mesh)
    profile = Profile(
        e_grid = energy_filter.values,
        x_grid = mesh_info.return_x_grid(),
        y_grid = mesh_info.return_y_grid(),
        z_grid = mesh_info.return_z_grid(),
        values = tally.mean.flatten(),
        std_dev = tally.std_dev.flatten(),
    )
    tally_shape = (
        len(profile.e_grid) - 1,
        len(profile.x_grid) - 1,
        len(profile.y_grid) - 1,
        len(profile.z_grid) - 1,
    )
    profile.values.shape = tally_shape
    profile.std_dev.shape = tally_shape
    return profile

def read_openmc_profiles(path:Path, identifiers:list[str]) -> Profiles:
    profiles = Profiles()
    sp = openmc.StatePoint(path)
    for identifier in identifiers:
        tally = sp.get_tally(name=identifier)
        profiles.profiles[identifier] = _read_openmc_profile(tally)
    return profiles


### MCNP Functions
mcnp_runtime_pattern = re.compile(r'computer time\s*=\s*([\d.]+)\s*(\w+)', re.IGNORECASE)
def read_mcnp_runtime(path:Path) -> float:
    with frb.FileReadBackwards(path, encoding="ascii") as file:
        for line in file:
            match = mcnp_runtime_pattern.search(line)
            if not match:
                continue
            value = float(match.group(1))
            units = match.group(2).lower()
            if units not in unit_to_seconds:
                raise ValueError(f"Unexpected time unit: {units}")
            return value * unit_to_seconds[units]
    raise RuntimeError(f"Unable to read runtime for {path}")

mcnp_keff_pattern = re.compile(r'final result\s+(\d+\.\d+)\s+(\d+\.\d+)', re.IGNORECASE)
def read_mcnp_k(path:Path) -> KResults:
    with frb.FileReadBackwards(path, encoding="ascii") as file:
        for line in file:
            match = mcnp_keff_pattern.search(line)
            if not match:
                continue
            k = float(match.group(1))
            std = float(match.group(2))
            return KResults(val=k, std=std)
    raise RuntimeError(f"Unable to read keff for {path}")

def read_mcnp_spectrum(path:Path, identifier:str) -> Spectrum:
    with h5.File(path, 'r') as file:
        try:
            tally = file[f'results/mesh_tally/{identifier}']
            # fmesh normalizes to per volume so undo that since I just want total
            del_x = tally['grid_x'][-1] - tally['grid_x'][0]
            del_y = tally['grid_y'][-1] - tally['grid_y'][0]
            del_z = tally['grid_z'][-1] - tally['grid_z'][0]
            vol = del_x*del_y*del_z
            # fmesh tallies always start with 0 energy and have the total at the end
            # Trim these off
            spectrum = Spectrum(
                bounds = np.array(tally['grid_energy'][1:])*10**6,
                values = np.array(tally['mean'][1:-1,0,0,0,0])*vol,
                std_dev = np.array(tally['relative_standard_error'][1:-1,0,0,0,0])*vol,
            )
            normalize_to_lethargy(spectrum)
            return spectrum
        except KeyError:
            raise RuntimeError(f"Unable to read/find tally {identifier} for {path}")

def _fix_e_grid(profile: Profile):
    if profile.e_grid[0] == profile.e_grid[1]:
        profile.e_grid = profile.e_grid[1:]
        profile.values = profile.values[1:,:,:,:]
        profile.std_dev = profile.std_dev[1:,:,:,:]

def _read_mcnp_profile(tally:h5.Group) -> Profile:
    # FMESH tallies are stored in [energy, time, z, y, z] indexing
    # Time binning is not needed so slice it out
    profile = Profile(
        e_grid = np.array(tally['grid_energy'][:])*10**6,
        x_grid = np.array(tally['grid_x'][:]),
        y_grid = np.array(tally['grid_y'][:]),
        z_grid = np.array(tally['grid_z'][:]),
        values = np.array(tally['mean'][:,0,:,:,:]),
        std_dev = np.array(tally['relative_standard_error'][:,0,:,:,:]),
    )
    _fix_e_grid(profile)
    return profile

def read_mcnp_profiles(path:Path, identifiers:list[str]) -> Profiles:
    profiles = Profiles()
    with h5.File(path, 'r') as file:
        for identifier in identifiers:
            tally = file[f'results/mesh_tally/{identifier}']
            profiles.profiles[identifier] = _read_mcnp_profile(tally)
    return profiles

### Functions
model_pattern = re.compile(r"^model\.*", re.IGNORECASE)
slurm_pattern = re.compile(r".*\.slurm$", re.IGNORECASE)
result_pattern = re.compile(r"^slurm-\d+\.(out|err)$", re.IGNORECASE)
def get_simulations(path: Path, accessor: CodeResultsAccessor) -> list[Simulation]:
    simulations = []
    for root, dirs, files in os.walk(path):
        root_path = Path(root).absolute()
        model_matches = [f for f in files if model_pattern.match(f)]
        slurm_matches = [f for f in files if slurm_pattern.match(f)]
        if model_matches and slurm_matches:
            was_run = any(result_pattern.match(f) for f in files)
            problem = root_path.parent.name
            run = root_path.name
            simulations.append(
                Simulation(
                    location=root_path,
                    problem=problem,
                    run=run,
                    model_file=model_matches[0],
                    slurm_file=slurm_matches[0],
                    was_run=was_run,
                    result_accessor=accessor,
                )
            )
    return simulations


### Globals
OPENMC: CodeResultsAccessor = CodeResultsAccessor(
    runtime_result_pattern = "statepoint*",
    runtime_normalization  = 1,
    runtime_reader         = read_openmc_runtime,

    k_eff_result_pattern = "statepoint*",
    k_eff_reader         = read_openmc_k,

    spectrum_result_pattern = "statepoint*",
    spectrum_reader         = read_openmc_spectrum,

    profiles_result_pattern = "statepoint*",
    profiles_reader         = read_openmc_profiles,
)

MCNP: CodeResultsAccessor = CodeResultsAccessor(
    runtime_result_pattern = "out*",
    runtime_normalization  = 1/11.0,
    runtime_reader         = read_mcnp_runtime,

    k_eff_result_pattern = "out*",
    k_eff_reader         = read_mcnp_k,

    spectrum_result_pattern = "runtp*",
    spectrum_reader         = read_mcnp_spectrum,

    profiles_result_pattern = "runtp*",
    profiles_reader         = read_mcnp_profiles,
)

CODE_ACCESSORS: dict[str, CodeResultsAccessor] = {
    "OpenMC": OPENMC,
    "MCNP": MCNP,
}
