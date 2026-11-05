from Basilisk.moduleTemplates import cModuleTemplate
from Basilisk.moduleTemplates import cppModuleTemplate
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros


def run():
    """
    Illustration of enabling and disabling tasks
    """

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess("dynamicsProcess")

    # create the dynamics task and specify the integration update time
    dynProcess.addTask(scSim.CreateNewTask("cTask", macros.sec2nano(1.)))
    dynProcess.addTask(scSim.CreateNewTask("cppTask", macros.sec2nano(1.)))

    # create modules
    mod1 = cModuleTemplate.cModuleTemplate()
    mod1.ModelTag = "cModule1"
    scSim.AddModelToTask("cTask", mod1)

    mod2 = cppModuleTemplate.CppModuleTemplate()
    mod2.ModelTag = "cppModule2"
    scSim.AddModelToTask("cppTask", mod2)

    #  initialize Simulation:
    scSim.InitializeSimulation()

    # execute BSK for a single step
    scSim.TotalSim.SingleStepProcesses()

    dynProcess.disableAllTasks()
    print("all tasks disabled")
    scSim.TotalSim.SingleStepProcesses()
    print("BSK executed a single simulation step")

    scSim.enableTask("cTask")
    scSim.TotalSim.SingleStepProcesses()
    print("BSK executed a single simulation step")

    scSim.enableTask("cppTask")
    scSim.TotalSim.SingleStepProcesses()
    print("BSK executed a single simulation step")

    scSim.disableTask("cppTask")
    scSim.TotalSim.SingleStepProcesses()
    print("BSK executed a single simulation step")

    return


if __name__ == "__main__":
    run()