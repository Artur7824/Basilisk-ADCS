import os
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging

from Basilisk.utilities import (
    SimulationBaseClass, 
    macros,
    RigidBodyKinematics as rbk,  # Para conversão em Quatérnio (MRP2EP)
    orbitalMotion,
    simHelpers,  # Para conversão da matriz de inércia (np2EigenMatrix3d)
    simIncludeGravBody,
    simSetPlanetEnvironment,
)

from Basilisk.simulation import (
    GravityGradientEffector,
    dragDynamicEffector,
    extForceTorque,
    msisAtmosphere,
    simpleNav,
    spacecraft,
    zeroWindModel,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,  #Referencial Nadir
)

# configuração da simulação
simProcessName = "simProcess"
simTaskName = "simTask"

scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess(simProcessName)

simulationTimeStep = macros.sec2nano(0.1) #10Hz
dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

scSim.SetProgressBar(True)

# Definição do satélite
scObject = spacecraft.Spacecraft()
scObject.ModelTag = "cimatelite"

# Adiciona o satélite à tarefa principal
scSim.AddModelToTask(simTaskName, scObject)

# Propriedades Físicas (Inércia e Massa)

I = [0.033, 0.0,   0.0,
     0.0,   0.033, 0.0,
     0.0,   0.0,   0.0067]

scObject.hub.mHub = 4  # kg
scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)

# sugestão do notebook LM - condições iniciais
scObject.hub.sigma_BNInit = [[0.1], [-0.1], [0.2]]        # Atitude inicial em MRP
scObject.hub.omega_BN_BInit = [[0.01], [-0.01], [0.005]]  # rad/s de rotação inicial

# arrasto atmosférico
atmo = msisAtmosphere.MsisAtmosphere()
atmo.ModelTag = "MsisAtmo"

# Configuração dos Índices de Atividade Solar (F10.7 e Ap)
ap = 8
f107 = 110
sw_msg = {
        "ap_24_0": ap, "ap_3_0": ap, "ap_3_-3": ap, "ap_3_-6": ap, "ap_3_-9": ap,
        "ap_3_-12": ap, "ap_3_-15": ap, "ap_3_-18": ap, "ap_3_-21": ap, "ap_3_-24": ap,
        "ap_3_-27": ap, "ap_3_-30": ap, "ap_3_-33": ap, "ap_3_-36": ap, "ap_3_-39": ap,
        "ap_3_-42": ap, "ap_3_-45": ap, "ap_3_-48": ap, "ap_3_-51": ap, "ap_3_-54": ap,
        "ap_3_-57": ap, "f107_1944_0": f107, "f107_24_-24": f107}

swMsgList = []
for c, val in enumerate(sw_msg.values()):
    swMsgData = messaging.SwDataMsgPayload(dataValue=val)
    swMsgList.append(messaging.SwDataMsg().write(swMsgData))
    atmo.swDataInMsgs[c].subscribeTo(swMsgList[-1])

# Conecta o satélite ao modelo atmosférico
atmo.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, atmo)

# Drag Dynamic Effector
projArea = 0.03 #m²
dragCoeff = 2.2

dragEffector = dragDynamicEffector.DragDynamicEffector()
dragEffector.ModelTag = "DragEff"
dragEffector.coreParams.projectedArea = projArea
dragEffector.coreParams.dragCoeff = dragCoeff

dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])

scObject.addDynamicEffector(dragEffector)
scSim.AddModelToTask(simTaskName, dragEffector)

# definição da órbita com esféricos harmônicos
gravFactory = simIncludeGravBody.gravBodyFactory()

# Import dos caminhos de dados do Basilisk
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

# Cria a Terra com o modelo de harmônicos esféricos (Grau 2 inclui J2)
planet = gravFactory.createEarth()
ggm2b_j2_only_path = get_path(DataFile.LocalGravData.GGM2B_J2_only)
planet.useSphericalHarmonicsGravityModel(str(ggm2b_j2_only_path), 2)
planet.isCentralBody = True
mu = planet.mu

gravFactory.addBodiesTo(scObject)

#definição do vento
spiceObject = gravFactory.createSpiceInterface(
      time="2020 MAY 21 18:28:03 (UTC)",
      epochInMsg=True
        )
spiceObject.zeroBase = "Earth"
scSim.AddModelToTask(simTaskName, spiceObject, 10)
atmo.epochInMsg.subscribeTo(gravFactory.epochMsg) #notebook LM

windModel = zeroWindModel.ZeroWindModel()
windModel.ModelTag = "ZeroWind"

windModel.planetPosInMsg.subscribeTo(spiceObject.planetStateOutMsgs[0])
windModel.epochInMsg.subscribeTo(gravFactory.epochMsg)
windModel.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, windModel)

dragEffector.windVelInMsg.subscribeTo(windModel.envOutMsgs[0])

#gradiente de gravidade
ggEff = GravityGradientEffector.GravityGradientEffector()
ggEff.ModelTag = scObject.ModelTag
ggEff.addPlanetName(planet.planetName)
scObject.addDynamicEffector(ggEff)
scSim.AddModelToTask(simTaskName, ggEff)

rLEO = planet.radEquator + 500.0 * 1000.0
oe = orbitalMotion.ClassicElements()
oe.a = rLEO
oe.e = 0.0001
oe.i = 33.3 * macros.D2R
oe.Omega = 48.2 * macros.D2R
oe.omega = 347.8 * macros.D2R
oe.f = 85.3 * macros.D2R

rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit = rN
scObject.hub.v_CN_NInit = vN

#definição dos sensores de atitude
sNavObject = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNavigation"
scSim.AddModelToTask(simTaskName, sNavObject)

# setup hillPoint guidance module
attGuidance = hillPoint.hillPoint()
attGuidance.ModelTag = "hillPoint"
scSim.AddModelToTask(simTaskName, attGuidance)

# setup the attitude tracking error evaluation module
attError = attTrackingError.attTrackingError()
attError.ModelTag = "attErrorInertial3D"
scSim.AddModelToTask(simTaskName, attError)

attError.sigma_R0R = [0, 0, 0]

# Conecta o sensor de navegação aos estados reais do satélite
sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

attGuidance.transNavInMsg.subscribeTo(sNavObject.transOutMsg)

# Conecta o calculador de erro de atitude ao sensor e à referência Nadir
attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
attError.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)

# set the simulation time
n = np.sqrt(mu / oe.a / oe.a / oe.a)
P = 2.0 * np.pi / n

simulationTime = macros.sec2nano(3*P)

# Setup data logging before the simulation is initialized
numDataPoints = 400

samplingTime = simHelpers.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

# Grava Posição e Velocidade inerciais
snLog = sNavObject.transOutMsg.recorder(samplingTime)
scSim.AddModelToTask(simTaskName, snLog)

# Gravao Erro de Atitude
attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
scSim.AddModelToTask(simTaskName, attErrorLog)

scSim.InitializeSimulation()
scSim.ConfigureStopTime(simulationTime)
scSim.ExecuteSimulation()

timeData = snLog.times() * macros.NANO2MIN

# Dados orbitais
global posData_global, velData_global
posData_global = snLog.r_BN_N
velData_global = snLog.v_BN_N

# Dados de atitude e erro de orientação
global attErrSigma_global, attErrOmega_global
attErrSigma_global = attErrorLog.sigma_BR
attErrOmega_global = attErrorLog.omega_BR_B

# CONVERSÃO DE MRP PARA QUATÉRNIO DE ERRO (q_BR) - gemini
quatErr_global = np.array([rbk.MRP2EP(s) for s in attErrSigma_global])

# full gemini
# ==============================================================================
# PLOTAGEM DOS GRÁFICOS
# ==============================================================================
timeOrbits = (snLog.times() * macros.NANO2SEC) / P

plt.close("all")

# Figura 1: Quatérnios de Erro
plt.figure(1, figsize=(9, 5))
plt.plot(timeOrbits, quatErr_global[:, 0], label=r'$q_0$ (Escalar)', color='black', linewidth=1.5)
plt.plot(timeOrbits, quatErr_global[:, 1], label=r'$q_1$ (Eixo $x$)', color='tab:red')
plt.plot(timeOrbits, quatErr_global[:, 2], label=r'$q_2$ (Eixo $y$)', color='tab:green')
plt.plot(timeOrbits, quatErr_global[:, 3], label=r'$q_3$ (Eixo $z$)', color='tab:blue')
plt.title(r"Evolução do Quatérnio de Erro em Relação ao Nadir ($q_{B/R}$)")
plt.xlabel("Tempo [Órbitas]")
plt.ylabel("Componentes do Quatérnio")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.7)

# Figura 2: Erro de Velocidade Angular
plt.figure(2, figsize=(9, 5))
omega_deg = attErrOmega_global * macros.R2D
plt.plot(timeOrbits, omega_deg[:, 0], label=r'$\omega_{B/R, x}$', color='tab:red')
plt.plot(timeOrbits, omega_deg[:, 1], label=r'$\omega_{B/R, y}$', color='tab:green')
plt.plot(timeOrbits, omega_deg[:, 2], label=r'$\omega_{B/R, z}$', color='tab:blue')
plt.title(r"Erro de Velocidade Angular ($\omega_{B/R}$)")
plt.xlabel("Tempo [Órbitas]")
plt.ylabel("Velocidade Angular [deg/s]")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.7)

# Figura 3: Posição Orbital
plt.figure(3, figsize=(9, 5))
pos_km = posData_global / 1000.0
plt.plot(timeOrbits, pos_km[:, 0], label=r'$r_x$', color='tab:red')
plt.plot(timeOrbits, pos_km[:, 1], label=r'$r_y$', color='tab:green')
plt.plot(timeOrbits, pos_km[:, 2], label=r'$r_z$', color='tab:blue')
plt.title(r"Componentes da Posição Inercial do Satélite ($r_{BN/N}$)")
plt.xlabel("Tempo [Órbitas]")
plt.ylabel("Posição [km]")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()