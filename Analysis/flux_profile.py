import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass

from Modules.Simulations import get_simulations, CODE_ACCESSORS
from Modules.Simulations import SimulationResults, RunResult, Profiles, Profile

ASSEMBLY_PIN_MASK = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    )
CROCUS_UO2_MASK = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    ]
    )
CROCUS_UMET_MASK = np.array(
    [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
    )
RCF_PIN_MASK = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ]
    )

@dataclass
class TallyInfo:
    identifier: str
    mask: np.ndarray | None = None
    name: str        | None = None
    value_label: str | None = None
    x_label: str     | None = None
    y_label: str     | None = None

MCNP_A_T_1: TallyInfo = TallyInfo(
    identifier='mesh_tally_14', 
    name='Flux vs Position', 
    value_label='Relative Flux',
    )
MCNP_A_T_2: TallyInfo = TallyInfo(
    identifier='mesh_tally_24', 
    mask=ASSEMBLY_PIN_MASK,
    name='Heating',          
    value_label='Relative Heating',
    )
MCNP_C_T_1: TallyInfo = TallyInfo(
    identifier='mesh_tally_14', 
    name='Flux vs Position', 
    value_label='Relative Flux',
    )
MCNP_C_T_2: TallyInfo = TallyInfo(
    identifier='mesh_tally_24', 
    mask=CROCUS_UO2_MASK,
    name='UO2 Pin Flux',     
    value_label='Relative Flux',
    )
MCNP_C_T_3: TallyInfo = TallyInfo(
    identifier='mesh_tally_34', 
    mask=CROCUS_UMET_MASK,
    name='UMET Pin Flux',    
    value_label='Relative Flux',
    )
MCNP_R_T_1: TallyInfo = TallyInfo(
    identifier='mesh_tally_14', 
    name='Flux vs Position', 
    value_label='Relative Flux',
    )
MCNP_R_T_2: TallyInfo = TallyInfo(
    identifier='mesh_tally_24', 
    mask=RCF_PIN_MASK,
    name='Pin Flux',         
    value_label='Relative Flux',
    )

OPENMC_A_T_1: TallyInfo = TallyInfo(
    identifier='Flux vs Position', 
    name='Flux vs Position', 
    value_label='Relative Flux',
    )
OPENMC_A_T_2: TallyInfo = TallyInfo(
    identifier='Heating',          
    mask=ASSEMBLY_PIN_MASK,
    name='Heating',          
    value_label='Relative Heating',
    )
OPENMC_C_T_1: TallyInfo = TallyInfo(
    identifier='Flux vs Position', 
    name='Flux vs Position', 
    value_label='Relative Flux',
    )
OPENMC_C_T_2: TallyInfo = TallyInfo(
    identifier='UO2 Pin Flux',     
    mask=CROCUS_UO2_MASK,
    name='UO2 Pin Flux',     
    value_label='Relative Flux',
    )
OPENMC_C_T_3: TallyInfo = TallyInfo(
    identifier='UMET Pin Flux',    
    mask=CROCUS_UMET_MASK,
    name='UMET Pin Flux',    
    value_label='Relative Flux',
    )
OPENMC_R_T_1: TallyInfo = TallyInfo(
    identifier='Flux vs Position', 
    name='Flux vs Position', 
    value_label='Relative Flux',
    )
OPENMC_R_T_2: TallyInfo = TallyInfo(
    identifier='Pin Flux',         
    mask=RCF_PIN_MASK,
    name='Pin Flux',         
    value_label='Relative Flux',
    )

profile_tally_info = {
    "MCNP":{
        "Assembly": [
            MCNP_A_T_1,
            MCNP_A_T_2,
            ],
        "CROCUS": [
            MCNP_C_T_1,
            MCNP_C_T_2,
            MCNP_C_T_3,
            ],
        "RCF": [
            MCNP_R_T_1,
            MCNP_R_T_2,
            ],
    },
    "OpenMC":{
        "Assembly": [
            OPENMC_A_T_1,
            OPENMC_A_T_2,
            ],
        "CROCUS": [
            OPENMC_C_T_1,
            OPENMC_C_T_2,
            OPENMC_C_T_3,
            ],
        "RCF": [
            OPENMC_R_T_1,
            OPENMC_R_T_2,
            ],
    }
}

def _clean_z(z: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    result = np.sum(z, axis=-1)
    if mask is not None:
        result = np.where(mask == 1, result, np.nan)
        norm = np.nanmax(result)
    else:
        norm = np.max(result)
    if norm == 0 or np.isnan(norm):
        return result * 0
    return result / norm

def _draw_panel(ax, x, y, z, *, label, title=None):
    # plot = ax.pcolormesh(x, y, z.T, shading="flat", edgecolors='none')
    extent = (x.min(), x.max(), y.min(), y.max())
    plot = ax.imshow(z.T, extent=extent, origin="lower", interpolation="nearest")
    cbar = ax.figure.colorbar(plot, ax=ax, label=label)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel("Position [cm]", fontsize=12)
    ax.set_ylabel("Position [cm]", fontsize=12)
    ax.tick_params(labelsize=12)

def _plot_single_profile(data: Profile, save_loc: Path, title_start: str, tally_info: TallyInfo):
    save_loc = save_loc / f"{tally_info.name}.pdf"
    title = f"{title_start} {tally_info.name}"
    x = data.x_grid
    y = data.y_grid
    z = _clean_z(data.values[0], tally_info.mask)
    fig, ax = plt.subplots()
    _draw_panel(ax, x, y, z, label=tally_info.value_label, title=title)
    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)

def _plot_thermal_fast_profile(data: Profile, save_loc: Path, title_start: str, tally_info: TallyInfo): 
    save_loc = save_loc / f"{tally_info.name    }.pdf"
    fig_title = f"{title_start} {tally_info.name}"
    x = data.x_grid
    y = data.y_grid
    thermal = _clean_z(data.values[0], tally_info.mask)
    fast    = _clean_z(data.values[1], tally_info.mask)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    _draw_panel(ax1, x, y, thermal, label=tally_info.value_label, title="Thermal")
    _draw_panel(ax2, x, y, fast,    label=tally_info.value_label, title="Fast")
    fig.suptitle(fig_title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)

def _get_tally_info(profile_data: Profile, tally_list: list[TallyInfo]):
    tally_info = None
    for tally in tally_list:
        if tally.identifier == profile_data.identifier:
            tally_info = tally
            break
    if tally_info == None:
        raise RuntimeError(f"Unable to find tally info for {profile_data.identifier}")
    return tally_info

def plot(run_data: RunResult[Profiles], save_loc: Path, title_start: str, tally_list: list[TallyInfo]):
    for profile_data in run_data.data.profiles.values():
        tally_info = _get_tally_info(profile_data, tally_list)
        print(f"  {tally_info.name}")
        if len(profile_data.e_grid) == 2:
            _plot_single_profile(profile_data, save_loc, title_start, tally_info)
        elif len(profile_data.e_grid) == 3:
            _plot_thermal_fast_profile(profile_data, save_loc, title_start, tally_info)
        else:
            raise RuntimeError(f"{title_start} {profile_data.identifier} has bad length for energy grid.")
    pass

def make_plots(data: SimulationResults[Profiles], output_path: Path):
    for code, code_data in data.codes.items():
        for problem, problem_data in code_data.problems.items():
            for run, run_data in problem_data.runs.items():
                print(f"Working on {code} {problem} {run}!")
                save_loc = output_path / code / problem / run
                save_loc.mkdir(parents=True, exist_ok=True)
                title_start = f"{code} {problem} {run}"
                plot(run_data, save_loc, title_start, profile_tally_info[code][problem])

if __name__ == "__main__":
    parent = Path(".").absolute().parent
    output_parent = Path("Outputs")
    output_loc = output_parent / "Figures" / "Profiles"
    output_loc.mkdir(parents=True, exist_ok=True)
    # codes = ["MCNP", "OpenMC"]
    codes = ["MCNP"]
    data = SimulationResults[Profiles]()
    for code in codes:
        path = parent / code
        if not path.exists():
            continue
        accessor = CODE_ACCESSORS[code]
        simulations = get_simulations(path, accessor)
        for simulation in simulations:
            if not simulation.was_run:
                # Skip simulations that have not been run yet
                continue
            if simulation.problem.endswith("_no_tallies"):
                # Skip simulations that have do not have tallies
                continue
            identifiers = [tally.identifier for tally in profile_tally_info[code][simulation.problem]]
            data.add_result(code,
                            simulation.problem,
                            simulation.run, 
                            simulation.result_accessor.get_profiles(simulation.location, identifiers))
    make_plots(data, output_loc)