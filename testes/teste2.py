#esse modelo usa esféricos harmônicos até 10° ordem;
#esse modelo inclui o método de facets no aerodrag;
#esse modelo tem a modelagem do SRP ~ resolvida com o claude;

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    RigidBodyKinematics as rbk,  # Conversão MRP -> quatérnio (MRP2EP)
    orbitalMotion,
    simHelpers,                  # Para conversão da matriz de inércia (np2EigenMatrix3d)
    simIncludeGravBody,
)
from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

from Basilisk.simulation import (
    GravityGradientEffector,
    extForceTorque,
    msisAtmosphere,
    simpleNav,
    spacecraft,
    zeroWindModel,
    facetDragDynamicEffector,
    facetSRPDynamicEffector,
    eclipse,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,  # Referencial Nadir
)

#configuração da simulação
simProcessName = "simProcess"
simTaskName = "simTask"

scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess(simProcessName)

simulationTimeStep = macros.sec2nano(0.1)  # 10 Hz
dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

scSim.SetProgressBar(True)

#definição do satélite
scObject = spacecraft.Spacecraft()
scObject.ModelTag = "cimatelite"

hubMass = 4.0  # [kg]
scObject.hub.mHub = hubMass

# CubeSat 3U
hubWidthX = 0.1   # [m]
hubWidthY = 0.1   # [m]
hubHeightZ = 0.3  # [m]

hubAreaX = hubWidthY * hubHeightZ
hubAreaY = hubWidthX * hubHeightZ
hubAreaZ = hubWidthX * hubWidthY

hubHalfX = hubWidthX / 2.0   # 0.05 m
hubHalfY = hubWidthY / 2.0   # 0.05 m
hubHalfZ = hubHeightZ / 2.0  # 0.15 m

#inércia de um paralelepípedo retângulo sobre o centro de massa
hubI11 = hubMass / 12.0 * (hubWidthY ** 2 + hubHeightZ ** 2)
hubI22 = hubMass / 12.0 * (hubWidthX ** 2 + hubHeightZ ** 2)
hubI33 = hubMass / 12.0 * (hubWidthX ** 2 + hubWidthY ** 2)

I = [hubI11, 0.0, 0.0,
     0.0, hubI22, 0.0,
     0.0, 0.0, hubI33]

scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
scObject.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)

# Condições iniciais de atitude ~ sugestão do LM
scObject.hub.sigma_BNInit = [[0.1], [-0.1], [0.2]]  # MRP
scObject.hub.omega_BN_BInit = [
    [60.0 * macros.D2R],
    [40.0 * macros.D2R],
    [50.0 * macros.D2R],
]  # [rad/s]

#definição dos corpos celestes (Terra e Sol) e interface SPICE
gravFactory = simIncludeGravBody.gravBodyFactory()
bodies = gravFactory.createBodies('earth', 'sun')

planet = bodies['earth']
planet.isCentralBody = True
sun = bodies['sun']

# Esféricos harmônicos de 10° ordem
ggm03s_full_path = get_path(DataFile.LocalGravData.GGM03S)
planet.useSphericalHarmonicsGravityModel(str(ggm03s_full_path), 10)
mu = planet.mu

gravFactory.addBodiesTo(scObject)

# Interface SPICE 
spiceObject = gravFactory.createSpiceInterface(
    time="2020 MAY 21 18:28:03 (UTC)",
    epochInMsg=True,
)
spiceObject.zeroBase = "Earth"
scSim.AddModelToTask(simTaskName, spiceObject, 10)  # prioridade alta: roda antes

earthStateMsg = spiceObject.planetStateOutMsgs[0]  # Terra
sunStateMsg = spiceObject.planetStateOutMsgs[1]    # Sol

# Adiciona o satélite à tarefa principal
scSim.AddModelToTask(simTaskName, scObject)

# Atmosfera MSIS
atmo = msisAtmosphere.MsisAtmosphere()
atmo.ModelTag = "MsisAtmo"

# Índices de atividade solar (F10.7 e Ap)
ap = 8
f107 = 110
sw_msg = {
    "ap_24_0": ap, "ap_3_0": ap, "ap_3_-3": ap, "ap_3_-6": ap, "ap_3_-9": ap,
    "ap_3_-12": ap, "ap_3_-15": ap, "ap_3_-18": ap, "ap_3_-21": ap, "ap_3_-24": ap,
    "ap_3_-27": ap, "ap_3_-30": ap, "ap_3_-33": ap, "ap_3_-36": ap, "ap_3_-39": ap,
    "ap_3_-42": ap, "ap_3_-45": ap, "ap_3_-48": ap, "ap_3_-51": ap, "ap_3_-54": ap,
    "ap_3_-57": ap, "f107_1944_0": f107, "f107_24_-24": f107,
}

swMsgList = []  
for c, val in enumerate(sw_msg.values()):
    swMsgData = messaging.SwDataMsgPayload(dataValue=val)
    swMsgList.append(messaging.SwDataMsg().write(swMsgData))
    atmo.swDataInMsgs[c].subscribeTo(swMsgList[-1])

atmo.planetPosInMsg.subscribeTo(earthStateMsg)
atmo.epochInMsg.subscribeTo(gravFactory.epochMsg)
atmo.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, atmo)

# modelagem do vento ~ zerowind model
windModel = zeroWindModel.ZeroWindModel()
windModel.ModelTag = "ZeroWind"
windModel.planetPosInMsg.subscribeTo(earthStateMsg)
windModel.epochInMsg.subscribeTo(gravFactory.epochMsg)
windModel.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, windModel)

#Facets: Aerodrag e SRP
dragCoeff = 2.2
sigSpec = 0.5
sigDiff = 0.1

hubDrag = facetDragDynamicEffector.FacetDragDynamicEffector()
hubDrag.ModelTag = "hubDrag"

SRPEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
SRPEffector.ModelTag = "SRPEffector"
SRPEffector.setNumFacets(6)  


def addCubesatFacets(facetDrag, facetSRP):
    halfDims = [hubHalfX, hubHalfY, hubHalfZ]
    areas = [hubAreaX, hubAreaY, hubAreaZ]

    dcm_identity = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]

    for axis in (0, 1, 2):
        for sign in (+1.0, -1.0):
            normal_list = [0.0, 0.0, 0.0]
            normal_list[axis] = sign

            location_list = [0.0, 0.0, 0.0]
            location_list[axis] = sign * halfDims[axis]

            rot_list = [0.0, 0.0, 0.0]

            # Aerodrag (4 parâmetros)
            facetDrag.addFacet(areas[axis], dragCoeff,
                               np.array(normal_list), np.array(location_list))

            # SRP (7 parâmetros)
            facetSRP.addFacet(
                areas[axis],    # 1. Área [m^2]
                dcm_identity,   # 2. Matriz de orientação da faceta [3x3]
                normal_list,    # 3. Normal da faceta [1x3]
                rot_list,       # 4. Eixo de rotação [1x3]
                location_list,  # 5. Localização / centro de pressão [1x3]
                sigDiff,        # 6. Coeficiente difuso
                sigSpec,        # 7. Coeficiente especular
            )


addCubesatFacets(hubDrag, SRPEffector)

# Conexões do arrasto
hubDrag.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])
hubDrag.windVelInMsg.subscribeTo(windModel.envOutMsgs[0])
scObject.addDynamicEffector(hubDrag)
scSim.AddModelToTask(simTaskName, hubDrag)

# Conexões do SRP: apenas o Sol [1].
# facetSRPDynamicEffector NÃO possui sunEclipseInMsg -> não conectar eclipse.
SRPEffector.sunInMsg.subscribeTo(sunStateMsg)
scObject.addDynamicEffector(SRPEffector)
scSim.AddModelToTask(simTaskName, SRPEffector)

#definição do eclipse
eclipseObject = eclipse.Eclipse()
eclipseObject.ModelTag = "EarthEclipse"

eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)
eclipseObject.addPlanetToModel(earthStateMsg)  # corpo eclipsante: Terra (SPICE [0])
eclipseObject.sunInMsg.subscribeTo(sunStateMsg)  # fonte luminosa: Sol (SPICE [1])
scSim.AddModelToTask(simTaskName, eclipseObject)

#Gradiente de gravidade
ggEff = GravityGradientEffector.GravityGradientEffector()
ggEff.ModelTag = scObject.ModelTag
ggEff.addPlanetName(planet.planetName)
scObject.addDynamicEffector(ggEff)
scSim.AddModelToTask(simTaskName, ggEff)

# definições de órbita
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

#exibição do eclipse
eclipseLog = eclipseObject.eclipseOutMsgs[0].recorder(samplingTime)
scSim.AddModelToTask(simTaskName, eclipseLog)

# Execução
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

# Figura 4: Fator de Sombra (1 = Sol pleno, 0 = Umbra)
plt.figure(4, figsize=(9, 3.5))
plt.plot(timeOrbits, eclipseLog.shadowFactor, color='tab:orange')
plt.title("Fator de Sombra (Eclipse)")
plt.xlabel("Tempo [Órbitas]")
plt.ylabel("Shadow factor")
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()