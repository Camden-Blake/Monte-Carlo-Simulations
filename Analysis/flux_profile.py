from pathlib import Path

from Modules.Simulations import get_simulations, CODE_ACCESSORS
from Modules.Simulations import SimulationResults, Profiles

import pprint as pp

profile_tally_identifiers = {
    "MCNP":{
        "Assembly": [
            'mesh_tally_14',
            'mesh_tally_24',
            ],
        "CROCUS": [
            'mesh_tally_14',
            'mesh_tally_24',
            'mesh_tally_34',
            ],
        "RCF": [
            'mesh_tally_14',
            'mesh_tally_24',
            ],
    },
    "OpenMC":{
        "Assembly": [
            'Flux vs Position',
            # 'Heating',
            ],
        "CROCUS": [
            'Flux vs Position',
            'UO2 Pin Flux',
            'UMET Pin Flux',
            ],
        "RCF": [
            'Flux vs Position',
            'Pin Flux',
            ],
    }
}

if __name__ == "__main__":
    parent = Path(".").absolute().parent
    output_parent = Path("Outputs")
    output_loc = output_parent / "Figures" / "Profiles"
    output_loc.mkdir(parents=True, exist_ok=True)
    codes = ["MCNP", "OpenMC"]
    # codes = ["OpenMC"]
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
            identifiers = profile_tally_identifiers[code][simulation.problem]
            data.add_result(code,
                            simulation.problem,
                            simulation.run, 
                            simulation.result_accessor.get_profiles(simulation.location, identifiers))
    # pp.pprint(data)