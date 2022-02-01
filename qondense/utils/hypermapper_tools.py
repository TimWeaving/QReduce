import json
import numpy as np

def hypermapper_specs(num_generators):
    """ Here we may specify the hypermapper parameters, governing
    its behaviour. Writes cofspecs to a .json file to be read into
    the hypermapper.optimzer.optimize method
    """
    # classical objective function general hypermapper optimizer specs
    output_directory = "data/hypermapper"
    cofspecs = {}
    cofspecs["run_directory"] = output_directory
    cofspecs["application_name"] = "ngs_optimization"
    cofspecs["log_file"] = "hypermapper_logfile.log"
    cofspecs["optimization_objectives"] = ["objfnc_sum"]
    #cofspecs["print_best"] = False
    cofspecs["models"] = {"model": "gaussian_process"} #"random_forest"

    # set the optimization variable parameters
    q_vars = [f'q{i}' for i in range(num_generators)]
    cofspecs["input_parameters"] = {}
    # ordinal variable type will optimize over specified discrete values (+/-1 here)
    for var in q_vars:
        cofspecs["input_parameters"][var] = {   "parameter_type": 'ordinal',
                                                "values": [-1, +1]}
    # real variable type is continuous within specified bounds
    cofspecs["input_parameters"]['theta'] = {   "parameter_type": 'real',
                                                "values": [-np.pi, +np.pi]}
        
    with open(output_directory+"/ngs_calculator.json", "w") as cofspecs_file:
        json.dump(cofspecs, cofspecs_file, indent=4)