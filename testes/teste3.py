#esse modelo inclui um ruído estocástico ao aerodrag e SRP;

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging, sysModel

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    RigidBodyKinematics as rbk,  # Conversão MRP -> quatérnio (MRP2EP)
    orbitalMotion,
    simHelpers,                  # Para conversão da matriz de inércia (np2EigenMatrix3d)
    simIncludeGravBody,
    vizSupport,
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
    meanRevertingNoiseStateEffector,
    svIntegrators,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,  # Referencial Nadir
)

#gemini + LM + claude ~ conexão do ruído estocástico com facetdrag
class StochasticAtmoBridge(sysModel.SysModel):
    def __init__(self, scObject, atmoModule, stateName):
        super().__init__()
        self.ModelTag = "StochasticAtmoBridge"
        self.scObject = scObject
        self.atmoModule = atmoModule
        self.stateName = stateName
        self.stateObject = None

        self.stochAtmoPayload = messaging.AtmoPropsMsgPayload()
        self.stochAtmoMsg = messaging.AtmoPropsMsg()

    def Reset(self, CurrentSimNanos):
        # O objeto de estado é obtido no primeiro UpdateState, quando o
        # spacecraft já registrou todos os estados.
        self.stateObject = None
        # Escreve uma mensagem inicial para que a saída exista desde t = 0
        self.stochAtmoPayload = messaging.AtmoPropsMsgPayload()
        self.stochAtmoMsg.write(self.stochAtmoPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        if self.stateObject is None:
            self.stateObject = self.scObject.dynManager.getStateObject(self.stateName)

        # delta_rho: estado escalar atual do processo de Ornstein-Uhlenbeck
        delta_rho = float(np.array(self.stateObject.getState()).flatten()[0])

        # Densidade nominal do MSIS [kg/m^3]
        nominalDensity = self.atmoModule.envOutMsgs[0].read().neutralDensity

        # rho_stoch = rho_nominal * (1 + delta_rho), limitada a valores não negativos
        self.stochAtmoPayload.neutralDensity = max(0.0, nominalDensity * (1.0 + delta_rho))

        self.stochAtmoMsg.write(self.stochAtmoPayload, CurrentSimNanos, self.moduleID)

#derivado diretamente do anterior
class StochasticSolarBridge(sysModel.SysModel):
    def __init__(self, scObject, sunStateMsg, stateName, nominalSolarFlux=1361.0):
        super().__init__()
        self.ModelTag = "StochasticSolarBridge"
        self.scObject = scObject
        self.sunStateMsg = sunStateMsg
        self.stateName = stateName
        self.nominalSolarFlux = nominalSolarFlux
        self.stateObject = None

        self.stochSolarPayload = messaging.SolarFluxMsgPayload()
        self.stochSolarMsg = messaging.SolarFluxMsg()

    def Reset(self, CurrentSimNanos):
        self.stateObject = None
        self.stochSolarPayload.flux = self.nominalSolarFlux
        self.stochSolarMsg.write(self.stochSolarPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        if self.stateObject is None:
            self.stateObject = self.scObject.dynManager.getStateObject(self.stateName)

        delta_solar = float(np.array(self.stateObject.getState()).flatten()[0])

        stoch_flux = max(0.0, self.nominalSolarFlux * (1.0 + delta_solar))
        self.stochSolarPayload.flux = stoch_flux

        self.stochSolarMsg.write(self.stochSolarPayload, CurrentSimNanos, self.moduleID)


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
scSim.AddModelToTask(simTaskName, spiceObject, ModelPriority=100)  # roda primeiro

earthStateMsg = spiceObject.planetStateOutMsgs[0]  # Terra
sunStateMsg = spiceObject.planetStateOutMsgs[1]    # Sol

# Adiciona o satélite à tarefa principal
scSim.AddModelToTask(simTaskName, scObject)

# SOSRA é o integrador forte recomendado para ruído aditivo ~ scenarioStochasticDragSpacecraft
integrator = svIntegrators.svStochasticIntegratorSOSRA(scObject)
scObject.setIntegrator(integrator)

# Atmosfera MSIS
atmo = msisAtmosphere.MsisAtmosphere()
atmo.ModelTag = "MsisAtmo"

# Índices de atividade solar (F10.7 e Ap)
ap = 8
f107 = 110
#atualização futura - usar dados reais de clima espacial
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
# Prioridade acima da ponte: a ponte precisa ler a densidade já atualizada do MSIS
scSim.AddModelToTask(simTaskName, atmo, ModelPriority=90)

# Variação estocástica da densidade
stochasticNoise = meanRevertingNoiseStateEffector.MeanRevertingNoiseStateEffector()
stochasticNoise.ModelTag = "StochasticExpAtmo"
stochasticNoise.setStationaryStd(0.15)       # desvio padrão estacionário de delta_rho
stochasticNoise.setTimeConstant(10 * 60.0)  # [s]
scObject.addStateEffector(stochasticNoise)

atmoBridge = StochasticAtmoBridge(scObject, atmo, stochasticNoise.getStateName())
scSim.AddModelToTask(simTaskName, atmoBridge, ModelPriority=80)

# modelagem do vento ~ zerowind model
windModel = zeroWindModel.ZeroWindModel()
windModel.ModelTag = "ZeroWind"
windModel.planetPosInMsg.subscribeTo(earthStateMsg)
windModel.epochInMsg.subscribeTo(gravFactory.epochMsg)
windModel.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, windModel)

#modelagem SRP
SRPEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
SRPEffector.ModelTag = "SRPEffector"
SRPEffector.setNumFacets(6) 

#ruído estocástico no fluxo solar
stochasticSolarNoise = meanRevertingNoiseStateEffector.MeanRevertingNoiseStateEffector()
stochasticSolarNoise.ModelTag = "StochasticSolarNoise"
stochasticSolarNoise.setStationaryStd(0.01)          # Desvio padrão de ~1% no fluxo solar
stochasticSolarNoise.setTimeConstant(10.0 * 60.0)     
scObject.addStateEffector(stochasticSolarNoise)

solarBridge = StochasticSolarBridge(
    scObject=scObject, 
    sunStateMsg=sunStateMsg, 
    stateName=stochasticSolarNoise.getStateName(),
    nominalSolarFlux=1361.0  # Constante solar nominal [W/m^2]
)
scSim.AddModelToTask(simTaskName, solarBridge, ModelPriority=85)

#Facets: Aerodrag e SRP
dragCoeff = 2.2
sigSpec = 0.5
sigDiff = 0.1

hubDrag = facetDragDynamicEffector.FacetDragDynamicEffector()
hubDrag.ModelTag = "hubDrag"

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
hubDrag.atmoDensInMsg.subscribeTo(atmoBridge.stochAtmoMsg)
hubDrag.windVelInMsg.subscribeTo(windModel.envOutMsgs[0])
scObject.addDynamicEffector(hubDrag)
scSim.AddModelToTask(simTaskName, hubDrag)

#definição do eclipse
eclipseObject = eclipse.Eclipse()
eclipseObject.ModelTag = "EarthEclipse"

eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)
eclipseObject.addPlanetToModel(earthStateMsg)  # corpo eclipsante: Terra (SPICE [0])
eclipseObject.sunInMsg.subscribeTo(sunStateMsg)  # fonte luminosa: Sol (SPICE [1])
scSim.AddModelToTask(simTaskName, eclipseObject)

# Conexões do SRP
SRPEffector.sunEclipseInMsg.subscribeTo(eclipseObject.eclipseOutMsgs[0])
SRPEffector.sunInMsg.subscribeTo(sunStateMsg)
if hasattr(SRPEffector, "solarFluxInMsg"):
    SRPEffector.solarFluxInMsg.subscribeTo(solarBridge.stochSolarMsg)

scObject.addDynamicEffector(SRPEffector)
scSim.AddModelToTask(simTaskName, SRPEffector, ModelPriority=70)

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

#full gemini

# 1. CONFIGURAÇÃO E REGISTRO DE DADOS 
numDataPoints = 1000
samplingTime = simHelpers.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

# Registradores de Navegação e Atitude
snLog = sNavObject.transOutMsg.recorder(samplingTime)
attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
scSim.AddModelToTask(simTaskName, snLog)
scSim.AddModelToTask(simTaskName, attErrorLog)

# Registradores de Ambiente, Pontes Estocásticas e Eclipse
deterministicAtmoLog = atmo.envOutMsgs[0].recorder(samplingTime)
stochasticAtmoLog = atmoBridge.stochAtmoMsg.recorder(samplingTime)
solarLog = solarBridge.stochSolarMsg.recorder(samplingTime)
eclipseLog = eclipseObject.eclipseOutMsgs[0].recorder(samplingTime)

scSim.AddModelToTask(simTaskName, deterministicAtmoLog)
scSim.AddModelToTask(simTaskName, stochasticAtmoLog)
scSim.AddModelToTask(simTaskName, solarLog)
scSim.AddModelToTask(simTaskName, eclipseLog)

# Registradores Únicos para os Efetores Dinâmicos
dragLog = hubDrag.logger(["forceExternal_B", "torqueExternalPntB_B"], samplingTime)
srpLog = SRPEffector.logger(["forceExternal_B", "torqueExternalPntB_B"], samplingTime)
ggLog = ggEff.logger(["torqueExternalPntB_B"], samplingTime)

scSim.AddModelToTask(simTaskName, dragLog)
scSim.AddModelToTask(simTaskName, srpLog)
scSim.AddModelToTask(simTaskName, ggLog)

# 2. DEFINIÇÃO DO ARQUICO DE VISUALIZAÇÃO

viz = vizSupport.enableUnityVisualization(
    scSim, 
    simTaskName, 
    scObject,
    saveFile="sim3U_CubeSat.bin")

# 3. EXECUÇÃO DA SIMULAÇÃO
scSim.InitializeSimulation()
scSim.ConfigureStopTime(simulationTime)
scSim.ExecuteSimulation()

# 4. EXTRAÇÃO E PROCESSAMENTO DOS DADOS
# Tempo em órbitas
timeOrbits = (snLog.times() * macros.NANO2SEC) / P

# Posição, altitude e variáveis de atitude
posData_global = snLog.r_BN_N
attErrSigma_global = attErrorLog.sigma_BR
attErrOmega_global = attErrorLog.omega_BR_B
quatErr_global = np.array([rbk.MRP2EP(s) for s in attErrSigma_global])

pos_matrix = np.array(posData_global).reshape(-1, 3)
r_mag = np.linalg.norm(pos_matrix, axis=1)
altitude_km = (r_mag - planet.radEquator) / 1000.0

# Variáveis estocásticas
rho_deterministic = np.array(deterministicAtmoLog.neutralDensity)
rho_stochastic = np.array(stochasticAtmoLog.neutralDensity)
solar_flux_data = np.array(solarLog.flux)

# Magnitudes de Força [N] e Torque [N*m]
magForceDrag = np.linalg.norm(dragLog.forceExternal_B, axis=1)
magForceSRP  = np.linalg.norm(srpLog.forceExternal_B, axis=1)

magTorqueDrag = np.linalg.norm(dragLog.torqueExternalPntB_B, axis=1)
magTorqueSRP  = np.linalg.norm(srpLog.torqueExternalPntB_B, axis=1)
magTorqueGG   = np.linalg.norm(ggLog.torqueExternalPntB_B, axis=1)

# 5. PLOTAGEM ESTRUTURADA E ORGANIZADA
plt.close("all")

# ------------------------------------------------------------------------------
# FIGURA 1: Análise de Atitude e Erro de Orientação
# ------------------------------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

# Subplot 1.1: Quatérnio de Erro Nadir
ax1.plot(timeOrbits, quatErr_global[:, 0], label=r'$q_0$ (Escalar)', color='black', linewidth=1.2)
ax1.plot(timeOrbits, quatErr_global[:, 1], label=r'$q_1$ (Eixo $x$)', color='tab:red')
ax1.plot(timeOrbits, quatErr_global[:, 2], label=r'$q_2$ (Eixo $y$)', color='tab:green')
ax1.plot(timeOrbits, quatErr_global[:, 3], label=r'$q_3$ (Eixo $z$)', color='tab:blue')
ax1.set_title(r"Evolução do Quatérnio de Erro em Relação ao Nadir ($q_{B/R}$)")
ax1.set_ylabel("Componentes $q$")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)

# Subplot 1.2: Erro de Velocidade Angular
omega_deg = attErrOmega_global * macros.R2D
ax2.plot(timeOrbits, omega_deg[:, 0], label=r'$\omega_{x}$', color='tab:red')
ax2.plot(timeOrbits, omega_deg[:, 1], label=r'$\omega_{y}$', color='tab:green')
ax2.plot(timeOrbits, omega_deg[:, 2], label=r'$\omega_{z}$', color='tab:blue')
ax2.set_title(r"Erro de Velocidade Angular ($\omega_{B/R}$)")
ax2.set_xlabel("Tempo [Órbitas]")
ax2.set_ylabel("Velocidade Angular [deg/s]")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 2: Trajetória Orbital e Variáveis do Ambiente Estocástico
# ------------------------------------------------------------------------------
fig2, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=False)

# Subplot 2.1: Posição Inercial
pos_km = posData_global / 1000.0
axs[0].plot(timeOrbits, pos_km[:, 0], label=r'$r_x$', color='tab:red')
axs[0].plot(timeOrbits, pos_km[:, 1], label=r'$r_y$', color='tab:green')
axs[0].plot(timeOrbits, pos_km[:, 2], label=r'$r_z$', color='tab:blue')
axs[0].set_title(r"Componentes da Posição Inercial ($r_{BN/N}$)")
axs[0].set_ylabel("Posição [km]")
axs[0].legend(loc="upper right")
axs[0].grid(True, linestyle="--", alpha=0.6)

# Subplot 2.2: Densidade Atmosférica vs Altitude
axs[1].semilogy(altitude_km[1:], rho_stochastic[1:], label="Stochastic Bridge", color='tab:blue', linewidth=1.2)
axs[1].semilogy(altitude_km[1:], rho_deterministic[1:], label="MSIS Nominal", color='tab:gray', linestyle='--', linewidth=1.2)
axs[1].set_title(r"Variabilidade Estocástica da Densidade Atmosférica ($\rho$)")
axs[1].set_xlabel("Altitude [km]")
axs[1].set_ylabel(r"$\rho$ [kg/m$^3$]")
axs[1].legend(loc="upper right")
axs[1].grid(True, which="both", linestyle="--", alpha=0.6)

# Subplot 2.3: Fluxo Solar Estocástico
axs[2].plot(timeOrbits, solar_flux_data, color='tab:orange', linewidth=1.2)
axs[2].axhline(1361.0, color='black', linestyle='--', label='Fluxo Nominal (1361 W/m²)')
axs[2].set_title(r"Variabilidade Estocástica do Fluxo Solar ($S_{stoch}$)")
axs[2].set_ylabel(r"Fluxo [$W/m^2$]")
axs[2].legend(loc="upper right")
axs[2].grid(True, linestyle="--", alpha=0.6)

# Subplot 2.4: Fator de Sombra (Eclipse)
axs[3].plot(timeOrbits, eclipseLog.shadowFactor, color='tab:orange', linewidth=1.2)
axs[3].set_title("Fator de Sombra Orbital (1 = Sol Pleno, 0 = Umbra)")
axs[3].set_xlabel("Tempo [Órbitas]")
axs[3].set_ylabel("Fator de Sombra")
axs[3].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 3: Magnitudes das Forças Perturbativas
# ------------------------------------------------------------------------------
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# Subplot 3.1: Força de Arrasto
ax1.semilogy(timeOrbits[1:], magForceDrag[1:], color='tab:blue', linewidth=1.2)
ax1.set_title(r"Magnitude da Força de Arrasto Aerodinâmico ($|F_{drag}|$)")
ax1.set_ylabel("Força [N]")
ax1.grid(True, which="both", linestyle="--", alpha=0.6)

# Subplot 3.2: Força de SRP
ax2.semilogy(timeOrbits[1:], magForceSRP[1:], color='tab:orange', linewidth=1.2)
ax2.set_title(r"Magnitude da Força de Pressão de Radiação Solar ($|F_{SRP}|$)")
ax2.set_xlabel("Tempo [Órbitas]")
ax2.set_ylabel("Força [N]")
ax2.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 4: Magnitudes dos Torques Perturbativos
# ------------------------------------------------------------------------------
fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

# Subplot 4.1: Torque de Arrasto
ax1.semilogy(timeOrbits[1:], magTorqueDrag[1:], color='tab:red', linewidth=1.2)
ax1.set_title(r"Torque de Arrasto Aerodinâmico ($|L_{drag}|$)")
ax1.set_ylabel("Torque [N·m]")
ax1.grid(True, which="both", linestyle="--", alpha=0.6)

# Subplot 4.2: Torque de SRP
ax2.semilogy(timeOrbits[1:], magTorqueSRP[1:], color='tab:orange', linewidth=1.2)
ax2.set_title(r"Torque de Pressão de Radiação Solar ($|L_{SRP}|$)")
ax2.set_ylabel("Torque [N·m]")
ax2.grid(True, which="both", linestyle="--", alpha=0.6)

# Subplot 4.3: Torque de Gradiente de Gravidade
ax3.semilogy(timeOrbits[1:], magTorqueGG[1:], color='tab:blue', linewidth=1.2)
ax3.set_title(r"Torque de Gradiente de Gravidade ($|L_{GG}|$)")
ax3.set_xlabel("Tempo [Órbitas]")
ax3.set_ylabel("Torque [N·m]")
ax3.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()

plt.show()