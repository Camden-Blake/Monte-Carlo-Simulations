from pathlib import Path

from Modules.Simulations import get_simulations, CODE_ACCESSORS
from Modules.Simulations import nested_dict, to_dict

def write_latex_tables(data: dict, output_path: Path):
    with open(output_path, "w") as f:
        for code, problems in data.items():
            f.write("\\begin{table}[]\n")
            f.write("\\begin{tabular}{|c|cc|c|l}\n")
            f.write("\\cline{1-4}\n")
            f.write(f"\\multirow{{2}}{{*}}{{{code}}} & \\multicolumn{{2}}{{c|}}{{Runtime [s]}} & & \\\\ \\cline{{2-4}}\n")
            f.write(" & \\multicolumn{1}{c|}{Standard} & OTF & Slowdown & \\\\ \\cline{1-4}\n")

            for problem, runs in sorted(problems.items()):
                std = runs.get("Standard", None)
                otf = runs.get("OTF", None)
                if std is None or otf is None:
                    continue
                slowdown = otf / std if std != 0 else float('inf')
                display_name = problem.replace("_no_tallies", " No Tallies")
                f.write(f"{display_name} & \\multicolumn{{1}}{{c|}}{{{std:.3f}}} & {otf:.3f} & {slowdown:.3f} & \\\\ \\cline{{1-4}}\n")

            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

if __name__ == "__main__":
    parent = Path(".").absolute().parent
    output_parent = Path("Outputs")
    output_loc = output_parent / "runtime_tables.tex"
    codes = ["MCNP", "OpenMC"]
    data = nested_dict()
    for code in codes:
        path = parent / code
        if not path.exists():
            continue
        accessor = CODE_ACCESSORS[code]
        simulations = get_simulations(path, accessor)
        for simulation in simulations:
            data[code][simulation.problem][simulation.run] = simulation.result_accessor.get_runtime(simulation.location)

    data = to_dict(data)
    write_latex_tables(data, output_loc) 
