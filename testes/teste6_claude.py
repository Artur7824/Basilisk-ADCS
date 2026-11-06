#esse modelo usa o seguinte pipeline:
'''
- primeiro, usa a lei de controle b-dot para estabilização inicial;
- executa 4 órbitas;
- configura o UKF, que usa as medições do star tracker como chute inicial,o filtro usa as medições do star tracker e do giroscópio para determinar atitude;
- executa 3 órbitas;
'''

'''
CORREÇÕES (teste6 -> corrigido) ~ Claude.
 Resumo:
  [1] Bloco 1 só com bdotTask; cadeia UKF/mrpFeedback/RW só em nominalTask, com prioridades na ordem do fluxo de dados
  [2] bias do giroscópio calibrado (GyroBiasCalibrator agora instanciado) e subtraído na ponte IMU->UKF
  [3] malha de controle fecha no UKF (attError lê o UKF); erro "verdade" separado (attErrorTruth) só para log
  [4] hot-start do UKF com P0/Q coerentes, dcm_BdyGyrpltf explícito e placeholders válidos em t=0
  [5] ganho V->torque = u_max/VMax; transição alinhada a tick de 0.1 s; recorders em tarefas sempre ativas
  [6] pós-processamento: alinhamento de sinal de quatérnios, amostragem de 2 s, verificação de tamanhos
'''

import os

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging, sysModel

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    RigidBodyKinematics as rbk,  # Conversão MRP -> quatérnio (MRP2EP)
    orbitalMotion,
    simHelpers,                 
    simIncludeGravBody,
    vizSupport,
    fswSetupRW,
    simIncludeRW,
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
    magneticFieldWMM,
    magnetometer,
    reactionWheelStateEffector,
    motorVoltageInterface,
    MtbEffector,
    starTracker,
    boreAngCalc,
    imuSensor,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint, 
    mrpFeedback,
    inertial3D,
    rwMotorTorque,
    rwMotorVoltage,
    tamComm,
    mtbMomentumManagement,
    inertialUKF,
)

# ==============================================================================
# PARÂMETROS DE ALTO NÍVEL
# ==============================================================================
N_ORBITS_BDOT = 4.0        # duração do Bloco 1 [órbitas]
N_ORBITS_TOTAL = 7.0       # duração total [órbitas]
LOG_PERIOD_S = 2.0         # [s] período dos recorders (múltiplo de 0.2 s)
GYRO_BIAS_DEG = 5.0        # [deg/s] bias do giroscópio (eixo x; y=-0.7x; z=0.5x)
GYRO_BIAS_TAU_S = 300.0    # [s] constante de tempo da calibração do bias
V_MAX = 3.5                # [V] tensão máxima das RWs (Hyperion RW210)
ENABLE_VIZ = True          # False para rodar sem o Vizard
VIZ_MODEL_PATH = r"C:\Users\azeve\Downloads\basilisk\cubesat3u_model\cimatelite.glb"


def set_task(sim, taskName, enabled):
    '''Liga/desliga uma tarefa inteira (com fallback caso a API mude de nome).'''
    fn = getattr(sim, "enableTask" if enabled else "disableTask", None)
    if fn is not None:
        fn(taskName)
        return
    for t in sim.TaskList:
        if t.Name == taskName:
            t.TaskData.enable() if enabled else t.TaskData.disable()
            return
    raise RuntimeError(f"Tarefa '{taskName}' não encontrada")


def zero_rw_speeds(sc, rwEff, nRW):
    '''Zera o estado de velocidade das rodas. O nome do estado pode levar sufixo de ID
    (reactionWheelOmegas<ID>), por isso tenta o atributo do efetor antes do nome fixo.'''
    for nome in (getattr(rwEff, "nameOfReactionWheelOmegasState", None), "reactionWheelOmegas"):
        if not nome:
            continue
        try:
            so = sc.dynManager.getStateObject(nome)
            so.setState(np.zeros((nRW, 1)).tolist())
            return True
        except Exception:
            continue
    print("[AVISO] Não consegui zerar o estado das RWs (nome do estado não encontrado).")
    return False


def mrp_angle_deg(sigma):
    '''Ângulo de rotação [deg] de MRPs (N,3), tratando o conjunto-sombra (|s|>1).'''
    n = np.linalg.norm(sigma, axis=1)
    n = np.where(n > 1.0, 1.0 / np.maximum(n, 1e-12), n)
    return 4.0 * np.degrees(np.arctan(n))


def align_sign(q_ref, q):
    '''q e -q são a mesma atitude: alinha o sinal de q ao de q_ref (para plot/erro).'''
    sg = np.sign(np.sum(q_ref * q, axis=1))
    sg[sg == 0] = 1.0
    return q * sg[:, None]


class BdotControl(sysModel.SysModel):
    #m_cmd = -K * dB/dt

    def __init__(self, kGain, maxDipole, filterTau=0.5):
        super().__init__()
        self.ModelTag = "BdotControl"
        self.kGain = kGain              # [A*m^2 / (T/s)]
        self.maxDipole = maxDipole      # [A*m^2] por eixo
        self.filterTau = filterTau      # [s] constante de tempo do filtro de dB/dt

        self.tamInMsg = messaging.TAMSensorBodyMsgReader()
        self.dipoleCmdOutMsg = messaging.MTBCmdMsg()

        self.lastMagField = None
        self.lastTimeNanos = None
        self.bDotFiltered = np.zeros(3)

    def Reset(self, CurrentSimNanos):
        self.lastMagField = None
        self.lastTimeNanos = None
        self.bDotFiltered = np.zeros(3)
        self.dipoleCmdOutMsg.write(messaging.MTBCmdMsgPayload(), CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        currentMagField = np.array(self.tamInMsg().tam_B)  # [T], referencial do corpo
        cmdPayload = messaging.MTBCmdMsgPayload()

        if self.lastMagField is not None:
            dt = (CurrentSimNanos - self.lastTimeNanos) * macros.NANO2SEC
            if dt > 0.0:
                bDot = (currentMagField - self.lastMagField) / dt

                # Filtro passa-baixa de 1ª ordem
                alpha = dt / (self.filterTau + dt)
                self.bDotFiltered += alpha * (bDot - self.bDotFiltered)

                # Lei B-dot
                dipole = -self.kGain * self.bDotFiltered
                dipole = np.clip(dipole, -self.maxDipole, self.maxDipole)

                cmd = np.zeros(len(cmdPayload.mtbDipoleCmds))
                cmd[:3] = dipole
                cmdPayload.mtbDipoleCmds = cmd.tolist()

        self.lastMagField = currentMagField
        self.lastTimeNanos = CurrentSimNanos

        self.dipoleCmdOutMsg.write(cmdPayload, CurrentSimNanos, self.moduleID)


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

#sugestão do claude para remover o bias do gyro ~ filtro média móvel
class GyroBiasCalibrator(sysModel.SysModel):
    def __init__(self, imuSensorObj, tau=300.0):
        super().__init__()
        self.ModelTag = "GyroBiasCalibrator"
        self.imuSensorObj = imuSensorObj
        self.tau = tau
        self.bias = np.zeros(3)
        self._init = False
        self._lastNanos = None

    def Reset(self, CurrentSimNanos):
        self.bias = np.zeros(3)
        self._init = False
        self._lastNanos = None

    def UpdateState(self, CurrentSimNanos):
        w = np.array(self.imuSensorObj.sensorOutMsg.read().AngVelPlatform)
        if not self._init:
            self.bias = w.copy()
            self._init = True
        else:
            dt = (CurrentSimNanos - self._lastNanos) * macros.NANO2SEC
            a = dt / (self.tau + dt)
            self.bias += a * (w - self.bias)
        self._lastNanos = CurrentSimNanos

#converte o IMUSensorMsgPayload do IMU para AccData do UKF
class ImuToGyrBuffBridge(sysModel.SysModel):
    # biasCorrection_B [rad/s] é subtraído do gyro: zero no Bloco 1, = bias calibrado no Bloco 2.
    # (o inertialUKF NÃO tem estado de bias: bias não removido vira "velocidade real")
    def __init__(self, imuSensorObj):
        super().__init__()
        self.ModelTag = "ImuToGyrBuffBridge"
        self.imuSensorObj = imuSensorObj
        self.accDataMsg = messaging.AccDataMsg()
        self.biasCorrection_B = np.zeros(3)

    def Reset(self, CurrentSimNanos):
        self.accDataMsg.write(messaging.AccDataMsgPayload(), CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        imuData = self.imuSensorObj.sensorOutMsg.read()
        pkt = messaging.AccPktDataMsgPayload()
        pkt.measTime = int(CurrentSimNanos)
        pkt.gyro_B = (np.array(imuData.AngVelPlatform) - self.biasCorrection_B).tolist()

        accDataPayload = messaging.AccDataMsgPayload()
        accDataPayload.accPkts = [pkt]
        self.accDataMsg.write(accDataPayload, CurrentSimNanos, self.moduleID)

#igual o de cima, mas pro ST
class StToUkfBridge(sysModel.SysModel):
    def __init__(self, stSensorObj, dcm_CB_matrix):
        super().__init__()
        self.ModelTag = "StToUkfBridge"
        self.stSensorObj = stSensorObj
        self.dcm_CB_matrix = np.array(dcm_CB_matrix)
        self.stAttMsg = messaging.STAttMsg()

    def Reset(self, CurrentSimNanos):
        stAttPayload = messaging.STAttMsgPayload()
        self.stAttMsg.write(stAttPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        stData = self.stSensorObj.sensorOutMsg.read()
        stAttPayload = messaging.STAttMsgPayload()
        
        # 1. Pega o quatérnio medido pelo ST (q_C/N)
        quat_CN = stData.qInrtl2Case
        
        # 2. Converte a atitude do referencial da câmera (C) para o referencial do corpo (B)
        quat_BC = rbk.C2EP(self.dcm_CB_matrix.T)
        quat_BN = rbk.addEP(quat_CN, quat_BC)
        
        # 3. Converte de Quatérnio (EP) para MRP
        sigma_BN = rbk.EP2MRP(quat_BN)
        
        # 4. Atribui os campos corretos da STAttMsgPayload
        stAttPayload.timeTag = stData.timeTag
        stAttPayload.MRP_BdyInrtl = sigma_BN
        
        self.stAttMsg.write(stAttPayload, CurrentSimNanos, self.moduleID)

#configuração da simulação
simProcessName = "simProcess"
simTaskName = "simTask"
tamTaskName = "100hzTask"
stTaskName = "5hzTask" 
fswTaskName = "fswTask"
bdotTaskName = "bdotTask"    
nomTaskName = "nominalTask"       

scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess(simProcessName)

simulationTimeStep = macros.sec2nano(0.1) # aumentar para 1kHz
tamTimeStep = macros.sec2nano(0.1)        # aumentar para 100Hz
fswTimeStep = macros.sec2nano(0.1)        # manter em 10Hz
stTimeStep = macros.sec2nano(0.2)         # manter em 5Hz

# Prioridade de tarefa (maior executa primeiro): dinâmica -> sensores -> FSW sempre ativa -> B-dot -> nominal
dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep), 400)
dynProcess.addTask(scSim.CreateNewTask(tamTaskName, tamTimeStep), 300)
dynProcess.addTask(scSim.CreateNewTask(stTaskName, stTimeStep), 250)
dynProcess.addTask(scSim.CreateNewTask(fswTaskName, fswTimeStep), 200)      # SEMPRE ATIVA
dynProcess.addTask(scSim.CreateNewTask(bdotTaskName, fswTimeStep), 190)     # Bloco 1
dynProcess.addTask(scSim.CreateNewTask(nomTaskName, fswTimeStep), 180)      # Bloco 2

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

# Condições iniciais
scObject.hub.sigma_BNInit = [[0.8], [-1.2], [1.0]]  # MRP
scObject.hub.omega_BN_BInit = [
    [60.0 * macros.D2R],
    [40.0 * macros.D2R],
    [50.0 * macros.D2R],]  # [rad/s]

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
scSim.AddModelToTask(simTaskName, scObject, ModelPriority=1)

# SOSRA é o integrador forte recomendado para ruído aditivo ~ scenarioStochasticDragSpacecraft
integrator = svIntegrators.svStochasticIntegratorSOSRA(scObject)
scObject.setIntegrator(integrator)

#Definição das RW ~ Hyperion RW210
torque_max = 0.0001  # [N.m] Torque máximo

rwFactory = simIncludeRW.rwFactory()

varRWModel = messaging.BalancedWheels
#varRWModel = messaging.JitterSimple

RW1 = rwFactory.create('custom', [1, 0, 0], maxMomentum=0.006, Omega=0.0  # RPM inicial
                       , Omega_max = 15000.0 #RPM
                       , u_max = torque_max
                       , rWB_B=[0, 0, 0]  # meters
                       , RWModel=varRWModel)

RW2 = rwFactory.create('custom', [0, 1, 0], maxMomentum=0.006, Omega=0.0  # RPM inicial
                       , Omega_max = 15000.0 #RPM
                       , u_max = torque_max
                       , rWB_B=[0, 0, 0]  # meters
                       , RWModel=varRWModel)

RW3 = rwFactory.create('custom', [0, 0, 1], maxMomentum=0.006, Omega=0.0  # RPM inicial
                       , Omega_max = 15000.0 #RPM
                       , u_max = torque_max
                       , rWB_B=[0, 0, 0]  # meters
                       , RWModel=varRWModel)

numRW = rwFactory.getNumOfDevices()
rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
rwStateEffector.ModelTag = "RW_cluster"
rwFactory.addToSpacecraft(rwStateEffector.ModelTag, rwStateEffector, scObject)

scSim.AddModelToTask(simTaskName, rwStateEffector, ModelPriority=2)

gain_rw210 = torque_max / V_MAX  # [N.m/V]: tensão máxima <-> torque máximo (antes: /3.3 com VMax=3.5 => +6% de ganho)

rwVoltageIO = motorVoltageInterface.MotorVoltageInterface()
rwVoltageIO.ModelTag = "rwVoltageInterface"
rwVoltageIO.setGains(np.array([gain_rw210] * numRW))

scSim.AddModelToTask(simTaskName, rwVoltageIO)

#Campo magnético WMM25
magModule = magneticFieldWMM.MagneticFieldWMM()
wmm_path = get_path(DataFile.MagneticFieldData.WMM)
magModule.configureWMMFile(str(wmm_path))
magModule.ModelTag = "WMM"

magModule.addSpacecraftToModel(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, magModule)

magModule.epochInMsg.subscribeTo(gravFactory.epochMsg)

#3-axis magnetometer ~ MPU9250
TAM = magnetometer.Magnetometer()
TAM.ModelTag = "TAM_sensor"
TAM.scaleFactor = 1.0

TAM.senNoiseStd = [600e-9, 600e-9, 600e-9] # 0.6µT
TAM.senBias = [0, 0, -1e-6]  # copiado do example do basilisk
TAM.maxOutput = 4.8e-3   #  4800µT
TAM.minOutput = -4.8e-3  # -4800µT

tamCommObj = tamComm.tamComm()
tamCommObj.dcm_BS = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
tamCommObj.ModelTag = "tamComm"

TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
TAM.magInMsg.subscribeTo(magModule.envOutMsgs[0])
tamCommObj.tamInMsg.subscribeTo(TAM.tamDataOutMsg)

scSim.AddModelToTask(tamTaskName, TAM)
scSim.AddModelToTask(simTaskName, tamCommObj)

#Magnetorquer ~ Hyperion HT-MTQ200
mtbEff = MtbEffector.MtbEffector()
mtbEff.ModelTag = "MtbEff"
scObject.addDynamicEffector(mtbEff)
scSim.AddModelToTask(simTaskName, mtbEff)

mtbMomentumManagementObj = mtbMomentumManagement.mtbMomentumManagement()
mtbMomentumManagementObj.wheelSpeedBiases = [1000. * macros.rpm2radsec, 1000. * macros.rpm2radsec,
                                             1000. * macros.rpm2radsec]
mtbMomentumManagementObj.cGain = 0.003
mtbMomentumManagementObj.ModelTag = "mtbMomentumManagement"
# (adicionado à nominalTask mais abaixo, com prioridade após o rwMotorTorque)

mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
mtbConfigParams.numMTB = 3
maxDipole = 1.0
mtbConfigParams.GtMatrix_B =[1.0, 0., 0.,
                             0., 1.0, 0.,
                             0., 0., 1.0]

mtbConfigParams.maxMtbDipoles = [maxDipole]*mtbConfigParams.numMTB
mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)

#Star Tracker
ST = starTracker.StarTracker()
ST.ModelTag = "StarTracker"

dcm_CB_matrix = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0]])

ST.dcm_CB = dcm_CB_matrix.tolist()

# ST-200
arcsec2rad = (np.pi / 180.0) / 3600.0
stdCorrectionFactor = 1.5  # Fator de ajuste "padrão"

# Desvio padrão 1-sigma
sigma_pitch_yaw = (30.0 / 3.0) * arcsec2rad   # ~4.85e-5 rad (Pitch e Yaw)
sigma_roll      = (200.0 / 3.0) * arcsec2rad  # ~3.23e-4 rad (Roll)

P_x = (sigma_pitch_yaw * stdCorrectionFactor) ** 2
P_y = (sigma_pitch_yaw * stdCorrectionFactor) ** 2
P_z = (sigma_roll * stdCorrectionFactor) ** 2

ST.PMatrix = [
    [P_x, 0.0, 0.0],
    [0.0, P_y, 0.0],
    [0.0, 0.0, P_z]]

walkBound_transversal = 60.0 * arcsec2rad   # ~2.91e-4 rad
walkBound_boresight   = 400.0 * arcsec2rad  # ~1.94e-3 rad

ST.walkBounds = [
    [walkBound_transversal], 
    [walkBound_transversal], 
    [walkBound_boresight]]

ST.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(stTaskName, ST)

#Cone de visão do Star Tracker

#Sol
stSunBAC = boreAngCalc.BoreAngCalc()
stSunBAC.ModelTag = "starTrackerSunBoresight"
stSunBAC.boreVec_B = [0., -1., 0.] 
stSunBAC.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
stSunBAC.celBodyInMsg.subscribeTo(sunStateMsg)
scSim.AddModelToTask(simTaskName, stSunBAC)

# terra
stEarthBAC = boreAngCalc.BoreAngCalc()
stEarthBAC.ModelTag = "starTrackerEarthBoresight"
stEarthBAC.boreVec_B = [0., -1., 0.] 
stEarthBAC.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
stEarthBAC.celBodyInMsg.subscribeTo(earthStateMsg)
scSim.AddModelToTask(simTaskName, stEarthBAC)

#sensor IMU ~ Gyro - MPU9250
ImuSensor = imuSensor.ImuSensor()
ImuSensor.ModelTag = "imusensor"
ImuSensor.sensorPos_B = [0.0, 0.0, 0.0]

yaw = 0.0 # [rad]
pitch = 0.0  # [rad]
roll = 0.0 # [rad]

ImuSensor.setBodyToPlatformDCM(yaw, pitch, roll) 

ImuSensor.senRotMax = 2000.0 * macros.D2R

#bias máximo de 5°
gyro_bias = GYRO_BIAS_DEG * macros.D2R #bias máximo
ImuSensor.senRotBias = [gyro_bias,-gyro_bias*0.7,gyro_bias/2]
gyro_bias_true_deg = np.array([GYRO_BIAS_DEG, -0.7*GYRO_BIAS_DEG, 0.5*GYRO_BIAS_DEG])

gyro_std = 0.1 * macros.D2R 
ImuSensor.PMatrixGyro = [[gyro_std, 0.0,      0.0],
                         [0.0,      gyro_std, 0.0],
                         [0.0,      0.0,      gyro_std]]

walk_gyro = 0.6 * macros.D2R 
ImuSensor.setErrorBoundsGyro(np.array([walk_gyro, walk_gyro, walk_gyro]))

ImuSensor.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, ImuSensor)

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
sigSpec = 0.068 #referente aos materiais da placa solar ~ GaInP/GaAs/Ge on Ge substrate
sigDiff = 0.012 #idem

hubDrag = facetDragDynamicEffector.FacetDragDynamicEffector()
hubDrag.ModelTag = "hubDrag"

#gemini
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

# set the simulation time
n = np.sqrt(mu / oe.a / oe.a / oe.a)
P = 2.0 * np.pi / n

# Transição e fim alinhados a um tick de 0.1 s (UKF.Reset/mrpFeedback.Reset usam este instante)
tempo_bdot = int(round(N_ORBITS_BDOT * P / 0.1)) * simulationTimeStep
totalSimTime = int(round(N_ORBITS_TOTAL * P / 0.1)) * simulationTimeStep
logDt = macros.sec2nano(LOG_PERIOD_S)

#definição B-dot
B_ref = 3.0e-5
J_min = min(hubI11, hubI22, hubI33)
kGainRef = 4.0 * np.pi * (1.0 + np.sin(oe.i)) * J_min / (P * B_ref ** 2)

kGain = kGainRef          # [A*m^2 / (T/s)];
bdotFilterTau = 0.5       # [s]

bdotControlObj = BdotControl(kGain=kGain, maxDipole=maxDipole, filterTau=bdotFilterTau)
bdotControlObj.tamInMsg.subscribeTo(tamCommObj.tamOutMsg)
scSim.AddModelToTask(bdotTaskName, bdotControlObj)

mtbEff.mtbCmdInMsg.subscribeTo(bdotControlObj.dipoleCmdOutMsg)
mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

#UKF
ukfObject = inertialUKF.inertialUKF()
ukfObject.ModelTag = "inertialUKF"
ukfObject.alpha = 0.01
ukfObject.beta = 2.0
ukfObject.kappa = 0.0

fswRwParamMsg = rwFactory.getConfigMessage()
vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

imuBridge = ImuToGyrBuffBridge(ImuSensor)
# sempre ativas (fswTask): ponte do gyro e calibrador de bias
scSim.AddModelToTask(fswTaskName, imuBridge)
biasCal = GyroBiasCalibrator(ImuSensor, tau=GYRO_BIAS_TAU_S)
scSim.AddModelToTask(fswTaskName, biasCal)

ukfObject.massPropsInMsg.subscribeTo(vcMsg)
ukfObject.gyrBuffInMsg.subscribeTo(imuBridge.accDataMsg)
ukfObject.rwParamsInMsg.subscribeTo(fswRwParamMsg)
ukfObject.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)

stBridge = StToUkfBridge(ST, dcm_CB_matrix)
scSim.AddModelToTask(stTaskName, stBridge)

ukfObject.STDatasStruct.numST = 1
ukfObject.STDatasStruct.STMessages[0].stInMsg.subscribeTo(stBridge.stAttMsg)
R_st_case = np.diag([P_x, P_y, P_z])
R_st_body = dcm_CB_matrix.T @ R_st_case @ dcm_CB_matrix
R_st_mrp = R_st_body / 16.0  # Aproximação de pequenas rotações MRP

ukfObject.STDatasStruct.STMessages[0].noise = R_st_mrp.flatten().tolist()

ukfObject.dcm_BdyGyrpltf = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]   # plataforma do gyro = corpo

# Placeholders válidos (não-nulos) para o Reset de t=0; valores reais aplicados na transição
ukfObject.stateInit = [0.0] * 6
ukfObject.covarInit = np.diag([1e-4] * 3 + [1e-4] * 3).flatten().tolist()
ukfObject.qNoise = np.diag([1e-8] * 3 + [1e-6] * 3).flatten().tolist()

# Hot-start (aplicado na transição)
P0_SIGMA = 1e-6                    # [MRP^2]  ST muito mais precisa que ~0.1 deg
P0_OMEGA = np.radians(0.5) ** 2    # [(rad/s)^2] resíduo do gyro após remover o bias (antes 1e-3 = 1.8 deg/s 1-sigma)
Q_SIGMA = 1e-8
Q_OMEGA = 1e-6
P0_ukf = np.diag([P0_SIGMA] * 3 + [P0_OMEGA] * 3)
Q_ukf = np.diag([Q_SIGMA] * 3 + [Q_OMEGA] * 3)

scSim.AddModelToTask(nomTaskName, ukfObject, ModelPriority=100)

#definição dos sensores de atitude
sNavObject = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNavigation"
scSim.AddModelToTask(fswTaskName, sNavObject)

# setup hillPoint guidance module
attGuidance = hillPoint.hillPoint()
attGuidance.ModelTag = "hillPoint"
scSim.AddModelToTask(fswTaskName, attGuidance)

# setup the attitude tracking error evaluation module
attError = attTrackingError.attTrackingError()
attError.ModelTag = "attError"
scSim.AddModelToTask(nomTaskName, attError, ModelPriority=90)

attError.sigma_R0R = [0, 1, 0]

# Conecta o sensor de navegação aos estados reais do satélite
sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
attGuidance.transNavInMsg.subscribeTo(sNavObject.transOutMsg)

# Conecta o calculador de erro de atitude ao sensor e à referência Nadir
attError.attNavInMsg.subscribeTo(ukfObject.navStateOutMsg)     # CONTROLE LÊ O UKF (antes: sNavObject = verdade)
attError.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)

# Erro "verdade" (apenas log/gráficos; não alimenta o controle). Sempre ativo.
attErrorTruth = attTrackingError.attTrackingError()
attErrorTruth.ModelTag = "attErrorTruth"
attErrorTruth.sigma_R0R = [0, 1, 0]
attErrorTruth.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
attErrorTruth.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)
scSim.AddModelToTask(fswTaskName, attErrorTruth)

# setup the MRP Feedback control module
mrpControl = mrpFeedback.mrpFeedback()
mrpControl.ModelTag = "mrpFeedback"
scSim.AddModelToTask(nomTaskName, mrpControl, ModelPriority=80)

mrpControl.K = 0.001    # Proporcional
mrpControl.P = 0.015    # Derivativo 
mrpControl.Ki = -1.0    # Integral 

fswRwParamMsg = rwFactory.getConfigMessage()
vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

#links
mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)

# add module that maps the Lr control torque into the RW motor torques
rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
rwMotorTorqueObj.ModelTag = "rwMotorTorque"
scSim.AddModelToTask(nomTaskName, rwMotorTorqueObj, ModelPriority=70)

#links
rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

# Make the RW control all three body axes
controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
rwMotorTorqueObj.controlAxes_B = controlAxes_B

fswRWVoltage = rwMotorVoltage.rwMotorVoltage()
fswRWVoltage.ModelTag = "rwMotorVoltage"

# Add test module to runtime call list
scSim.AddModelToTask(nomTaskName, fswRWVoltage, ModelPriority=50)

fswRWVoltage.VMin = 0.0   # piso/zona-morta positiva (não o limite inferior)
fswRWVoltage.VMax = V_MAX

#links MTB +RW
mtbMomentumManagementObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
mtbMomentumManagementObj.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbMomentumManagementObj.tamSensorBodyInMsg.subscribeTo(tamCommObj.tamOutMsg)
mtbMomentumManagementObj.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
mtbMomentumManagementObj.rwMotorTorqueInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
scSim.AddModelToTask(nomTaskName, mtbMomentumManagementObj, ModelPriority=60)

fswRWVoltage.torqueInMsg.subscribeTo(mtbMomentumManagementObj.rwMotorTorqueOutMsg)
fswRWVoltage.rwParamsInMsg.subscribeTo(fswRwParamMsg)
rwVoltageIO.motorVoltageInMsg.subscribeTo(fswRWVoltage.voltageOutMsg)
zeroRwCmdMsg = messaging.ArrayMotorTorqueMsg().write(messaging.ArrayMotorTorqueMsgPayload())
rwStateEffector.rwMotorCmdInMsg.subscribeTo(zeroRwCmdMsg)

# MTB começa comandado pelo B-dot (linha acima); religado ao momentum management na transição
mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

# ==============================================================================
# RECORDERS: todos criados ANTES do InitializeSimulation, em tarefas SEMPRE ativas e na
# mesma grade temporal (logDt). Evita vetores de tamanhos diferentes (ex.: 429 vs 1001).
# ==============================================================================
snLog = sNavObject.transOutMsg.recorder(logDt)
attNavLog = sNavObject.attOutMsg.recorder(logDt)
attErrorLog = attErrorTruth.attGuidOutMsg.recorder(logDt)        # verdade-terreno (antes: attError, inativo no Bloco 1)
ukfNavLog = ukfObject.navStateOutMsg.recorder(logDt)
rwCmdTorqueLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(logDt)
rwVoltLog = fswRWVoltage.voltageOutMsg.recorder(logDt)
for rec in (snLog, attNavLog, attErrorLog, ukfNavLog, rwCmdTorqueLog, rwVoltLog):
    scSim.AddModelToTask(fswTaskName, rec)

ukfFiltLog = None
try:
    ukfFiltLog = ukfObject.filtDataOutMsg.recorder(logDt)
    scSim.AddModelToTask(fswTaskName, ukfFiltLog)
except Exception:
    ukfFiltLog = None

deterministicAtmoLog = atmo.envOutMsgs[0].recorder(logDt)
stochasticAtmoLog = atmoBridge.stochAtmoMsg.recorder(logDt)
solarLog = solarBridge.stochSolarMsg.recorder(logDt)
eclipseLog = eclipseObject.eclipseOutMsgs[0].recorder(logDt)
magLog = magModule.envOutMsgs[0].recorder(logDt)
tamLog = TAM.tamDataOutMsg.recorder(logDt)
mtbLog = mtbEff.logger(["torqueExternalPntB_B"], logDt)
stLog = ST.sensorOutMsg.recorder(logDt)                          # antes: stTimeStep (~200k pontos)
stSunBACLog = stSunBAC.angOutMsg.recorder(logDt)
stEarthBACLog = stEarthBAC.angOutMsg.recorder(logDt)
imuLog = ImuSensor.sensorOutMsg.recorder(logDt)
rwSpeedLog = rwStateEffector.rwSpeedOutMsg.recorder(logDt)
scLog = scObject.scStateOutMsg.recorder(logDt)

for rec in (deterministicAtmoLog, stochasticAtmoLog, solarLog, eclipseLog, magLog, mtbLog,
            stSunBACLog, stEarthBACLog, imuLog, rwSpeedLog, scLog):
    scSim.AddModelToTask(simTaskName, rec)
scSim.AddModelToTask(tamTaskName, tamLog)
scSim.AddModelToTask(stTaskName, stLog)

dragLog = hubDrag.logger(["forceExternal_B", "torqueExternalPntB_B"], logDt)
srpLog = SRPEffector.logger(["forceExternal_B", "torqueExternalPntB_B"], logDt)
ggLog = ggEff.logger(["torqueExternalPntB_B"], logDt)
for rec in (dragLog, srpLog, ggLog):
    scSim.AddModelToTask(simTaskName, rec)

# 2. DEFINIÇÃO DO ARQUICO DE VISUALIZAÇÃO
if ENABLE_VIZ:
    viz = vizSupport.enableUnityVisualization(
        scSim,
        simTaskName,
        scObject,
        saveFile="sim3U_CubeSat.bin",
        rwEffectorList=rwStateEffector)

    viz.epochInMsg.subscribeTo(gravFactory.epochMsg)
    viz.settings.ambient = 0.8

    if os.path.exists(VIZ_MODEL_PATH):
        vizSupport.createCustomModel(
            viz,
            modelPath=VIZ_MODEL_PATH,
            simBodiesToModify=["cimatelite"],
            scale=[7.0, 7.0, 7.0])

    vizSupport.createConeInOut(
        viz, toBodyName='sun', coneColor='g',
        normalVector_B=[0.0, -1.0, 0.0], incidenceAngle=20 * macros.D2R,
        isKeepIn=True, coneHeight=5.0, coneName='ST200_FOV_Verde')

    vizSupport.createConeInOut(
        viz, toBodyName='sun', coneColor='r',
        normalVector_B=[0.0, -1.0, 0.0], incidenceAngle=20 * macros.D2R,
        isKeepIn=False, coneHeight=5.0, coneName='ST200_Sun_KeepOut')

#Bloco 1 ~ B-dot (RWs, UKF e mrpFeedback DESLIGADOS)
scSim.InitializeSimulation()

set_task(scSim, bdotTaskName, True)
set_task(scSim, nomTaskName, False)

scSim.ConfigureStopTime(tempo_bdot)
scSim.ExecuteSimulation()

# ==============================================================================
# TRANSIÇÃO (t = tempo_bdot)
# ==============================================================================
print("\n================ TRANSIÇÃO BLOCO 1 -> BLOCO 2 ================")
omega_true_deg_t = np.array(scObject.scStateOutMsg.read().omega_BN_B) * macros.R2D
print(f"|omega_real| na transição            : {np.linalg.norm(omega_true_deg_t):.3f} deg/s "
      f"(deve ser << {GYRO_BIAS_DEG} deg/s para a calibração do bias valer)")

# (1) bias do gyro calibrado com o satélite quase parado
bias_est = biasCal.bias.copy()
bias_est_deg = bias_est * macros.R2D
print(f"bias calibrado [deg/s]               : {np.round(bias_est_deg, 3)}")
print(f"bias verdadeiro (simulado) [deg/s]   : {np.round(gyro_bias_true_deg, 3)}")
print(f"erro de calibração [deg/s]           : {np.round(bias_est_deg - gyro_bias_true_deg, 3)}")
imuBridge.biasCorrection_B = bias_est

# (2) ST disponível? (apenas aviso; o hot-start usa a última medida)
sun_ang_t = np.degrees(stSunBAC.angOutMsg.read().missAngle)
earth_ang_t = np.degrees(stEarthBAC.angOutMsg.read().missAngle)
if sun_ang_t <= 40.0 or earth_ang_t <= 40.0:
    print(f"[AVISO] ST dentro de cone de exclusão na transição (Sol {sun_ang_t:.1f} deg, Terra {earth_ang_t:.1f} deg): "
          "hot-start pode usar medida inválida; considere adiar a transição.")

# (3) hot-start do UKF: x0 = [sigma_ST (já no corpo, via ponte), omega_gyro - bias]
sigma_BN_init = np.array(stBridge.stAttMsg.read().MRP_BdyInrtl)
omega_gyro = np.array(ImuSensor.sensorOutMsg.read().AngVelPlatform)
omega0 = omega_gyro - bias_est

ukfObject.stateInit = [float(x) for x in np.concatenate([sigma_BN_init, omega0])]
ukfObject.covarInit = P0_ukf.flatten().tolist()
ukfObject.qNoise = Q_ukf.flatten().tolist()
ukfObject.Reset(tempo_bdot)
mrpControl.Reset(tempo_bdot)

# (4) RWs: estado limpo em 0 RPM e só agora o comando do FSW é ligado
rw_before = np.array(rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds)[:numRW] / macros.RPM
print(f"RW antes do reset [RPM]              : {np.round(rw_before, 2)}")
zero_rw_speeds(scObject, rwStateEffector, numRW)

rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwVoltageIO.motorTorqueOutMsg)
mtbEff.mtbCmdInMsg.subscribeTo(mtbMomentumManagementObj.mtbCmdOutMsg)

# (5) troca de tarefas
set_task(scSim, bdotTaskName, False)
set_task(scSim, nomTaskName, True)
print("================================================================\n")

#BLOCO 2: UKF + mrpFeedback + MTB momentum management
scSim.ConfigureStopTime(totalSimTime)
scSim.ExecuteSimulation()

#pós-processamento ~ gemini (corrigido)

# ==============================================================================
# 5. PREPARAÇÃO DOS DADOS E PÓS-PROCESSAMENTO
# ==============================================================================
orb = lambda rec: (rec.times() * macros.NANO2SEC) / P

timeOrbits_sim = orb(snLog)
timeOrbits_fsw = orb(rwVoltLog)
timeOrbits_tam = orb(tamLog)
timeOrbits_st = orb(stLog)
timeOrbits_imu = orb(imuLog)
timeOrbits_ukf_log = orb(ukfNavLog)

# Checagem de consistência: todos os registradores na mesma grade (evita IndexError nas máscaras)
for nome, rec in [("rwVoltLog", rwVoltLog), ("rwCmdTorqueLog", rwCmdTorqueLog), ("ukfNavLog", ukfNavLog),
                  ("rwSpeedLog", rwSpeedLog), ("attErrorLog", attErrorLog), ("scLog", scLog),
                  ("attNavLog", attNavLog), ("imuLog", imuLog), ("stLog", stLog), ("tamLog", tamLog)]:
    if len(rec.times()) != len(snLog.times()):
        print(f"[AVISO] {nome}: {len(rec.times())} pontos != {len(snLog.times())} (snLog). "
              "Verifique se está em tarefa sempre ativa.")

# 1. Atitude real
sigma_BN_real = np.array(scLog.sigma_BN)
omega_real_rad = np.array(scLog.omega_BN_B)
quat_BN_real = np.array([rbk.MRP2EP(s_) for s_ in sigma_BN_real])
omega_BN_real_deg = omega_real_rad * macros.R2D
omega_true_norm = np.linalg.norm(omega_BN_real_deg, axis=1)

# 2. Erro de atitude (verdade em relação ao Nadir)
attErrSigma_global = attErrorLog.sigma_BR
attErrOmega_global = attErrorLog.omega_BR_B
quatErr_global = np.array([rbk.MRP2EP(s_) for s_ in attErrSigma_global])

# 3. Atuadores
dataOmegaRW = rwSpeedLog.wheelSpeeds[:, :numRW] / macros.RPM
dataVoltRW = rwVoltLog.voltage[:, :numRW]
dataTorqueMTB = mtbLog.torqueExternalPntB_B

# 4. Máscara de cegueira do ST
sun_angles_deg = stSunBACLog.missAngle * macros.R2D
earth_angles_deg = stEarthBACLog.missAngle * macros.R2D
sun_angles_interp = np.interp(stLog.times(), stSunBACLog.times(), sun_angles_deg)
earth_angles_interp = np.interp(stLog.times(), stEarthBACLog.times(), earth_angles_deg)
fov_threshold_deg = 40.0
st_blind_mask = (sun_angles_interp <= fov_threshold_deg) | (earth_angles_interp <= fov_threshold_deg)

# 5. Star Tracker: medido x real (com alinhamento de sinal q/-q)
quat_CN_measured = np.array(stLog.qInrtl2Case)
quat_BC = rbk.C2EP(dcm_CB_matrix.T)
quat_BN_measured = np.array([rbk.addEP(q_.tolist(), quat_BC) for q_ in quat_CN_measured])

quat_real_interp_st = np.zeros_like(quat_BN_measured)
for i_ in range(4):
    quat_real_interp_st[:, i_] = np.interp(stLog.times(), scLog.times(), quat_BN_real[:, i_])
quat_BN_measured = align_sign(quat_real_interp_st, quat_BN_measured)

error_arcsec = np.zeros((len(quat_BN_measured), 3))
for i_ in range(len(quat_BN_measured)):
    dq = np.array(rbk.subEP(quat_BN_measured[i_], quat_real_interp_st[i_]))
    error_arcsec[i_] = 2.0 * dq[1:] * (180.0 / np.pi) * 3600.0

quat_BN_measured_masked = np.copy(quat_BN_measured)
error_arcsec_masked = np.copy(error_arcsec)
quat_BN_measured_masked[st_blind_mask] = np.nan
error_arcsec_masked[st_blind_mask] = np.nan

# 6. IMU
omega_BN_measured_deg = np.array(imuLog.AngVelPlatform) * macros.R2D

# 7. UKF (sinal alinhado ao real: MRP2EP de conjuntos-sombra diferentes dá q e -q)
sigma_BN_ukf = np.array(ukfNavLog.sigma_BN)
quat_BN_ukf = np.array([rbk.MRP2EP(s_) for s_ in sigma_BN_ukf])
quat_BN_ukf = align_sign(quat_BN_real, quat_BN_ukf)
omega_ukf_deg = np.array(ukfNavLog.omega_BN_B) * macros.R2D

# Erros do UKF (mesma grade temporal dos logs => sem interpolação)
mask_ukf = timeOrbits_ukf_log >= N_ORBITS_BDOT + 0.01
att_err_ukf_deg = mrp_angle_deg(np.array([rbk.subMRP(a, b) for a, b in
                                          zip(sigma_BN_ukf[mask_ukf], sigma_BN_real[mask_ukf])]))
omega_err_ukf_deg = omega_ukf_deg[mask_ukf] - omega_BN_real_deg[mask_ukf]

# Máscaras
m_b1_sim = timeOrbits_sim < (N_ORBITS_BDOT - 1e-3)
m_b2_sim = timeOrbits_sim >= N_ORBITS_BDOT
m_b1_fsw = timeOrbits_fsw < (N_ORBITS_BDOT - 1e-3)
m_b2_fsw = timeOrbits_fsw >= N_ORBITS_BDOT

# ==============================================================================
# IMPRESSÃO DO DIAGNÓSTICO DE DESEMPENHO NO TERMINAL
# ==============================================================================
i_after = np.argmax(m_b2_sim)
print("\n===================== DIAGNÓSTICO =====================")
print("BLOCO 1 (0 -> 4 órbitas)")
print(f"   ||omega|| inicial / final (real)    : {omega_true_norm[0]:.2f} / {omega_true_norm[m_b1_sim][-1]:.3f} deg/s")
print(f"   max |Omega_RW|  (esperado ~ 0-10)   : {np.max(np.abs(dataOmegaRW[m_b1_sim])):.2f} RPM")
print(f"   max |V_RW|      (esperado = 0)      : {np.max(np.abs(dataVoltRW[m_b1_fsw])):.4f} V")
print("BLOCO 2 (4 -> 7 órbitas)")
print(f"   RW logo após a transição            : {np.round(dataOmegaRW[i_after], 2)} RPM")
print(f"   max |Omega_RW|                      : {np.max(np.abs(dataOmegaRW[m_b2_sim])):.0f} RPM (saturação = 15000)")
print(f"   max |V_RW|                          : {np.max(np.abs(dataVoltRW[m_b2_fsw])):.2f} V (limite {V_MAX})")
if att_err_ukf_deg.size:
    n_last = max(1, att_err_ukf_deg.size // 3)
    print(f"   erro de atitude do UKF (RMS, última 1/3): {np.sqrt(np.mean(att_err_ukf_deg[-n_last:]**2)):.4f} deg")
    print(f"   erro de atitude do UKF (máx)           : {np.max(att_err_ukf_deg):.4f} deg")
    print(f"   erro de omega do UKF (RMS, última 1/3) : {np.sqrt(np.mean(omega_err_ukf_deg[-n_last:]**2, axis=0)).round(4)} deg/s")
print(f"   ||omega|| real, média 4->7 órbitas  : {np.mean(omega_true_norm[m_b2_sim]):.3f} deg/s")
print("========================================================\n")

# ==============================================================================
# 6. PLOTAGEM ESTRUTURADA (APENAS AS 8 FIGURAS DO TESTE 7)
# ==============================================================================
plt.close("all")

# ------------------------------------------------------------------------------
# FIGURA 1: Erro de Atitude (Quatérnio de Erro em Relação ao Nadir) - verdade
# ------------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(timeOrbits_sim, quatErr_global[:, 0], label=r'$q_0$ (Escalar)', color='black', linewidth=1.2)
ax1.plot(timeOrbits_sim, quatErr_global[:, 1], label=r'$q_1$ (Eixo $x$)', color='tab:red', linewidth=1.0)
ax1.plot(timeOrbits_sim, quatErr_global[:, 2], label=r'$q_2$ (Eixo $y$)', color='tab:green', linewidth=1.0)
ax1.plot(timeOrbits_sim, quatErr_global[:, 3], label=r'$q_3$ (Eixo $z$)', color='tab:blue', linewidth=1.0)
ax1.axvline(N_ORBITS_BDOT, color='gray', linestyle=':')
ax1.set_title(r"Evolução do Quatérnio de Erro em Relação ao Nadir ($q_{B/R}$) - verdade")
ax1.set_xlabel("Tempo [Órbitas]")
ax1.set_ylabel("Componentes $q$")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 2: Erro de Velocidade Angular - verdade
# ------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 4))
omega_deg = attErrOmega_global * macros.R2D
ax2.plot(timeOrbits_sim, omega_deg[:, 0], label=r'$\omega_{x}$', color='tab:red')
ax2.plot(timeOrbits_sim, omega_deg[:, 1], label=r'$\omega_{y}$', color='tab:green')
ax2.plot(timeOrbits_sim, omega_deg[:, 2], label=r'$\omega_{z}$', color='tab:blue')
ax2.axvline(N_ORBITS_BDOT, color='gray', linestyle=':')
ax2.set_title(r"Erro de Velocidade Angular ($\omega_{B/R}$) - verdade")
ax2.set_xlabel("Tempo [Órbitas]")
ax2.set_ylabel("Velocidade Angular [deg/s]")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 3: Velocidade Angular das Rodas de Reação Hyperion RW210
# ------------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 4.5))
ax3.plot(timeOrbits_sim, dataOmegaRW[:, 0], label=r'$\Omega_1$ (Eixo $x$)', color='tab:red', linewidth=1.2)
ax3.plot(timeOrbits_sim, dataOmegaRW[:, 1], label=r'$\Omega_2$ (Eixo $y$)', color='tab:green', linewidth=1.2)
ax3.plot(timeOrbits_sim, dataOmegaRW[:, 2], label=r'$\Omega_3$ (Eixo $z$)', color='tab:blue', linewidth=1.2)
ax3.axhline(15000.0, color='black', linestyle='--', alpha=0.7, label=r'Saturação ($\pm 15.000$ RPM)')
ax3.axhline(-15000.0, color='black', linestyle='--', alpha=0.7)
ax3.axvline(N_ORBITS_BDOT, color='gray', linestyle=':')
ax3.set_title("Velocidade Angular das Rodas de Reação Hyperion RW210")
ax3.set_xlabel("Tempo [Órbitas]")
ax3.set_ylabel("Velocidade [RPM]")
ax3.legend(loc="upper right")
ax3.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 4: Torque Magnético Gerado pelo MTB Hyperion HT-MTQ200
# ------------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(9, 4.5))
ax4.plot(timeOrbits_sim, dataTorqueMTB[:, 0] * 1e6, label=r'$\tau_{MTB,x}$', color='tab:red', linewidth=1.0)
ax4.plot(timeOrbits_sim, dataTorqueMTB[:, 1] * 1e6, label=r'$\tau_{MTB,y}$', color='tab:green', linewidth=1.0)
ax4.plot(timeOrbits_sim, dataTorqueMTB[:, 2] * 1e6, label=r'$\tau_{MTB,z}$', color='tab:blue', linewidth=1.0)
ax4.axvline(N_ORBITS_BDOT, color='gray', linestyle=':')
ax4.set_title("Torque Magnético Gerado pelo MTB Hyperion HT-MTQ200")
ax4.set_xlabel("Tempo [Órbitas]")
ax4.set_ylabel(r"Torque [$\mu\text{N}\cdot\text{m}$]")
ax4.legend(loc="upper right")
ax4.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 5: Atitude: Real vs. Medida pelo Star Tracker
# ------------------------------------------------------------------------------
fig5, ax5 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
labels = [r'$q_0$ (Escalar)', r'$q_1$ (Eixo $x$)', r'$q_2$ (Eixo $y$)', r'$q_3$ (Eixo $z$)']
colors = ['black', 'tab:red', 'tab:green', 'tab:blue']

for i in range(4):
    ax5[i].plot(timeOrbits_sim, quat_BN_real[:, i], label='Real ($q_{B/N}$)', color='black', alpha=0.7, linewidth=1.2)
    ax5[i].plot(timeOrbits_st, quat_BN_measured_masked[:, i], label='Medido ST ($q_{B/N}$)', color=colors[i], linestyle='--', linewidth=1.0)
    ax5[i].fill_between(
        timeOrbits_st, 0, 1, where=st_blind_mask, color='red', alpha=0.25,
        transform=ax5[i].get_xaxis_transform(), label='ST inativo' if i == 0 else ""
    )
    ax5[i].set_ylabel(labels[i])
    ax5[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax5[i].legend(loc="upper right", ncol=3)

ax5[3].set_xlabel("Tempo [Órbitas]")
fig5.suptitle("Atitude: Real vs. Medida pelo Star Tracker", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 6: Ruído/Erro do Star Tracker
# ------------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(9, 4))
ax6.plot(timeOrbits_st, error_arcsec_masked[:, 0], label=r'Erro $x$', color='tab:red', alpha=0.8)
ax6.plot(timeOrbits_st, error_arcsec_masked[:, 1], label=r'Erro $y$', color='tab:green', alpha=0.8)
ax6.plot(timeOrbits_st, error_arcsec_masked[:, 2], label=r'Erro $z$', color='tab:blue', alpha=0.8)
ax6.fill_between(
    timeOrbits_st, 0, 1, where=st_blind_mask, color='red', alpha=0.25,
    transform=ax6.get_xaxis_transform(), label='ST Inativo'
)
ax6.set_title("Ruído/Erro do Star Tracker")
ax6.set_xlabel("Tempo [Órbitas]")
ax6.set_ylabel("Erro [arcsec]")
ax6.legend(loc="upper right")
ax6.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 7: Regime UKF (Órbitas 4 a 7): Velocidade Angular
# ------------------------------------------------------------------------------
m_imu = timeOrbits_imu >= N_ORBITS_BDOT
m_sim = timeOrbits_sim >= N_ORBITS_BDOT
m_ukf = timeOrbits_ukf_log >= N_ORBITS_BDOT

colors_w = ['tab:red', 'tab:green', 'tab:blue']
fig7, ax7 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
labels_w = [r'$\omega_x$ (Eixo $x$)' + '\n[deg/s]', r'$\omega_y$ (Eixo $y$)' + '\n[deg/s]', r'$\omega_z$ (Eixo $z$)' + '\n[deg/s]']

for i in range(3):
    ax7[i].plot(timeOrbits_sim[m_sim], omega_BN_real_deg[m_sim, i], label=r'Real ($\omega_{B/N}$)', color='black', linewidth=1.2)
    ax7[i].plot(timeOrbits_imu[m_imu], omega_BN_measured_deg[m_imu, i], label=r'IMU (com bias)', color='lightcoral' if i==0 else ('lightgreen' if i==1 else 'skyblue'), linestyle='--', alpha=0.8, linewidth=1.0)
    ax7[i].plot(timeOrbits_ukf_log[m_ukf], omega_ukf_deg[m_ukf, i], label=r'Estimado UKF', color='darkorange', linewidth=1.0)
    ax7[i].set_ylabel(labels_w[i])
    ax7[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax7[i].legend(loc="upper right", ncol=3)

ax7[2].set_xlabel("Tempo [Órbitas]")
fig7.suptitle("Regime UKF (Órbitas 4 a 7): Velocidade Angular", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 8: Estimativa de Atitude: Real (Pontilhada) vs. UKF (Contínua)
# ------------------------------------------------------------------------------
fig8, ax8 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
labels_q = [r'$q_0$ (Escalar)', r'$q_1$ (Eixo $x$)', r'$q_2$ (Eixo $y$)', r'$q_3$ (Eixo $z$)']
colors_q = ['black', 'tab:red', 'tab:green', 'tab:blue']

for i in range(4):
    ax8[i].plot(timeOrbits_sim[m_sim], quat_BN_real[m_sim, i], label='Atitude Real', color='black', linestyle=':', linewidth=2.0)
    ax8[i].plot(timeOrbits_ukf_log[m_ukf], quat_BN_ukf[m_ukf, i], label='Atitude Estimada (UKF)', color=colors_q[i], linewidth=1.2, alpha=0.85)
    ax8[i].set_ylabel(labels_q[i])
    ax8[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax8[i].legend(loc="upper right", ncol=2)

ax8[3].set_xlabel("Tempo [Órbitas]")
fig8.suptitle("Estimativa de Atitude: Real (Pontilhada) vs. UKF (Contínua)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 9 (extra): Desempenho do UKF (erro de atitude e de omega)
# ------------------------------------------------------------------------------
fig9, ax9 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t_u = timeOrbits_ukf_log[mask_ukf]
ax9[0].semilogy(t_u, np.maximum(att_err_ukf_deg, 1e-6), color='tab:purple')
ax9[0].set_ylabel("Erro de atitude do UKF [deg]")
ax9[0].grid(True, which="both", linestyle="--", alpha=0.5)
for i, c in enumerate(colors_w):
    ax9[1].plot(t_u, omega_err_ukf_deg[:, i], color=c, label=['x', 'y', 'z'][i], linewidth=1.0)
ax9[1].set_ylabel(r"Erro de $\omega$ do UKF [deg/s]")
ax9[1].set_xlabel("Tempo [Órbitas]")
ax9[1].legend(loc="upper right", ncol=3)
ax9[1].grid(True, linestyle="--", alpha=0.5)
fig9.suptitle("Desempenho do UKF (estimado - real)")
plt.tight_layout()

plt.show()

