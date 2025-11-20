from pathlib import Path

from Modules.Simulations import get_simulations, CODE_ACCESSORS
from Modules.Simulations import SimulationResults, Spectrum

import matplotlib.pyplot as plt
import itertools

def _plot_stairs_group(curves: dict[str, Spectrum], title: str, output_file: Path):
    linestyles = ['-', '--', '-.', ':']
    style_cycle = itertools.cycle(linestyles)
    for label, spectrum in curves.items():
        style = next(style_cycle)
        plt.stairs(spectrum.values, spectrum.bounds, label=label, linestyle=style)
    plt.title(title, fontsize=16)
    plt.xlabel("Energy [eV]", fontsize=14)
    plt.ylabel("Flux per Unit Lethargy [Arb. Units]", fontsize=14)
    plt.xscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()

def make_plots(data: SimulationResults[Spectrum], output_path: Path):
    for code, code_data in data.codes.items():
        for problem, problem_data in code_data.problems.items():
            run_curves = {run.name: run.data for run in problem_data.runs.values()}
            title = f"{code} {problem} Flux Spectrum"
            out = output_path / f"{code}_{problem}.pdf"
            _plot_stairs_group(run_curves, title=title, output_file=out)

        problem_curves = {
            problem: problem_data.runs['Standard'].data
            for problem, problem_data in code_data.problems.items()
            if "Standard" in problem_data.runs
        }
        title = f"{code} Flux Spectrum"
        out = output_path / f"{code}.pdf"
        _plot_stairs_group(problem_curves, title=title, output_file=out)

spectrum_tally_identifiers = {
    "MCNP":{
        "Assembly": 'mesh_tally_34',
        "CROCUS": 'mesh_tally_44',
        "RCF": 'mesh_tally_34',
    },
    "OpenMC":{
        "Assembly": 'Flux Spectrum',
        "CROCUS": 'Flux Spectrum',
        "RCF": 'Flux Spectrum',
    }
}

if __name__ == "__main__":
    parent = Path(".").absolute().parent
    output_parent = Path("Outputs")
    output_loc = output_parent / "Figures" / "Spectrums"
    output_loc.mkdir(parents=True, exist_ok=True)
    codes = ["MCNP", "OpenMC"]
    data = SimulationResults[Spectrum]()
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
            identifier = spectrum_tally_identifiers[code][simulation.problem]
            data.add_result(code,simulation.problem,simulation.run, simulation.result_accessor.get_spectrum(simulation.location, identifier))

    make_plots(data, output_loc)