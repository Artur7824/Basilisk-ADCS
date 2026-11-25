#nova lei de apontamento: 4 órbitas de b-dot -> 3 órbitas alternando entre b-dot e apontamento para uma lat/lon específica numa janela de visada
#lat/lon = senai

import os
from datetime import datetime, timezone

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
    groundLocation,
)

from Basilisk.fswAlgorithms import (
    attTrackingError,
    celestialTwoBodyPoint, 
    mrpFeedback,
    inertial3D,
    rwMotorTorque,
    rwMotorVoltage,
    tamComm,
    mtbMomentumManagement,
    inertialUKF,
    locationPointing,
)

# PARÂMETROS DE ALTO NÍVEL
N_ORBITS_BDOT = 5.0        # duração do b-dot
N_ORBITS_TOTAL = 7.5         # duração total
PASS_TARGET_ORBIT = 6.2    # órbita (contada de t=0) em que se quer a passagem sobre a estação
PASS_RAAN_OFFSET_DEG = 2.0 # desloca o rastro p/ passagem ~75° de elevação máx (menos taxa angular)
LOG_PERIOD_S = 2.0         # período dos recorders (múltiplo de 0.2 s)
GYRO_BIAS_DEG = 5.0        # bias do giroscópio (eixo x; y=-0.7x; z=0.5x)
GYRO_BIAS_TAU_S = 300.0    # [s] constante de tempo da calibração do bias
V_MAX = 3.5                # tensão máxima das RWs (Hyperion RW210)

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

class ResidualMagneticDipole(sysModel.SysModel):
    """Calcula tau_B = m_res_B x B_B e publica CmdTorqueBodyMsg para o ExtForceTorque."""
    def __init__(self, m_res_B):
        super().__init__()
        self.ModelTag = "ResidualMagneticDipole"
        self.m_res_B = np.array(m_res_B, dtype=float)  # [A*m^2] no referencial do corpo

        self.magInMsg = messaging.MagneticFieldMsgReader()
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.torqueOutMsg = messaging.CmdTorqueBodyMsg()

    def Reset(self, CurrentSimNanos):
        self.torqueOutMsg.write(messaging.CmdTorqueBodyMsgPayload(), CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        B_N = np.array(self.magInMsg().magField_N)         # [T] no referencial inercial N
        sigma_BN = np.array(self.scStateInMsg().sigma_BN)  # atitude B/N

        dcm_BN = rbk.MRP2C(sigma_BN)
        B_B = dcm_BN @ B_N                                 # projeta B para o corpo B
        tau_B = np.cross(self.m_res_B, B_B)                # [N*m]

        payload = messaging.CmdTorqueBodyMsgPayload()
        payload.torqueRequestBody = tau_B.tolist()
        self.torqueOutMsg.write(payload, CurrentSimNanos, self.moduleID)

#sugestão do gemini para remover o bias do gyro ~ filtro média móvel
class GyroBiasCalibrator(sysModel.SysModel):
    def __init__(self, imuSensorObj, tau=300.0):
        super().__init__()
        self.ModelTag = "GyroBiasCalibrator"
        self.imuSensorObj = imuSensorObj
        self.tau = tau
        self.bias = np.zeros(3)
        self._init = False
        self._lastNanos = None
        self.frozen = False      # congelado ao fim do detumbling: não aprende a rotação real do apontamento

    def Reset(self, CurrentSimNanos):
        self.bias = np.zeros(3)
        self._init = False
        self._lastNanos = None

    def UpdateState(self, CurrentSimNanos):
        if self.frozen:
            return
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

#igual o de cima, mas pro ST + desliga as leituras do ST na janela de oclusão
class StToUkfBridge(sysModel.SysModel):
    def __init__(self, stSensorObj, dcm_CB_matrix, sunBacObj=None, earthBacObj=None, fovThresholdDeg=40.0):
        super().__init__()
        self.ModelTag = "StToUkfBridge"
        self.stSensorObj = stSensorObj
        self.dcm_CB_matrix = np.array(dcm_CB_matrix)
        self.sunBacObj = sunBacObj
        self.earthBacObj = earthBacObj
        self.fovThresholdRad = fovThresholdDeg * macros.D2R
        self.stAttMsg = messaging.STAttMsg()

    def Reset(self, CurrentSimNanos):
        stAttPayload = messaging.STAttMsgPayload()
        self.stAttMsg.write(stAttPayload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        # 1. Verifica se o Star Tracker está cego por Sol ou Terra
        is_blind = False
        if self.sunBacObj is not None:
            sun_angle = self.sunBacObj.angOutMsg.read().missAngle
            if sun_angle <= self.fovThresholdRad:
                is_blind = True

        if self.earthBacObj is not None and not is_blind:
            earth_angle = self.earthBacObj.angOutMsg.read().missAngle
            if earth_angle <= self.fovThresholdRad:
                is_blind = True

        # 2. Se estiver cego, NÃO atualiza a atitude do ST enviando timeTag inválida/antiga
        if is_blind:
            # Ao mandar timeTag = 0 ou manter a mensagem inalterada,
            # o UKF detecta que não há nova medição e apenas propaga via IMU (dead-reckoning).
            return

        # 3. Leitura nominal quando o sensor está desobstruído
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

def st_is_blind():
    """True se o ST está dentro do cone de exclusão (Sol/Terra) agora: a mensagem da ponte ficaria velha."""
    sun_a = stSunBAC.angOutMsg.read().missAngle
    earth_a = stEarthBAC.angOutMsg.read().missAngle
    return (sun_a <= 40.0 * macros.D2R) or (earth_a <= 40.0 * macros.D2R)

def enter_pointing_mode(current_nanos):
    # 1. Hot-start do UKF: atitude do ST (via ponte) + omega do gyro - bias calibrado (congelado no fim do detumbling)
    sigma_BN_init = np.array(stBridge.stAttMsg.read().MRP_BdyInrtl)
    omega0 = np.array(ImuSensor.sensorOutMsg.read().AngVelPlatform) - imuBridge.biasCorrection_B
    ukfObject.stateInit = [float(x) for x in np.concatenate([sigma_BN_init, omega0])]
    ukfObject.covarInit = P0_ukf.flatten().tolist()
    ukfObject.qNoise = Q_ukf.flatten().tolist()
    ukfObject.Reset(current_nanos)
    attGuidance.Reset(current_nanos)     # zera o histórico do guidance (derivadas numéricas) após o salto de atitude do UKF
    mrpControl.Reset(current_nanos)

    # 2. Atuadores de apontamento fino
    zero_rw_speeds(scObject, rwStateEffector, numRW)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwVoltageIO.motorTorqueOutMsg)
    mtbEff.mtbCmdInMsg.subscribeTo(mtbMomentumManagementObj.mtbCmdOutMsg)

    # 3. Tarefas
    scSim.disableTask(bdotTaskName)
    scSim.enableTask(nomTaskName)

def enter_bdot_mode(current_nanos):
    # 1. Desacopla as RWs e zera as velocidades
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(zeroRwCmdMsg)
    zero_rw_speeds(scObject, rwStateEffector, numRW)

    # 2. Devolve os MTBs ao B-dot e reinicia o filtro derivativo
    bdotControlObj.Reset(current_nanos)
    mtbEff.mtbCmdInMsg.subscribeTo(bdotControlObj.dipoleCmdOutMsg)

    # 3. Tarefas
    scSim.disableTask(nomTaskName)
    scSim.enableTask(bdotTaskName)


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

simulationTimeStep = macros.sec2nano(0.1) # 1 kHz 
tamTimeStep = macros.sec2nano(0.1)        # 100 Hz 
fswTimeStep = macros.sec2nano(0.1)        # manter em 10Hz
stTimeStep = macros.sec2nano(0.2)         # manter em 5Hz

# Prioridades: dinâmica -> sensores -> FSW -> B-dot -> nominal
dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep), 400)
dynProcess.addTask(scSim.CreateNewTask(tamTaskName, tamTimeStep), 300)
dynProcess.addTask(scSim.CreateNewTask(stTaskName, stTimeStep), 250)
dynProcess.addTask(scSim.CreateNewTask(fswTaskName, fswTimeStep), 200)      
dynProcess.addTask(scSim.CreateNewTask(bdotTaskName, fswTimeStep), 190)     # Bloco 1
dynProcess.addTask(scSim.CreateNewTask(nomTaskName, fswTimeStep), 180)      # Bloco 2

scSim.SetProgressBar(False) 

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

#senai cimatec
lat_deg = -12.938326767568624
lon_deg = -38.387244122792886
alt_m   = 0.0  # altitude em metros acima do raio definido

# definições de órbita
rLEO = planet.radEquator + 500.0 * 1000.0
oe = orbitalMotion.ClassicElements()
oe.a = rLEO
oe.e = 0.0001
oe.i = 33.3 * macros.D2R
oe.omega = 347.8 * macros.D2R
oe.f = 85.3 * macros.D2R

#sugestão do claude para garantir janela de comunicação entre 4-7 órbitas
EPOCH_UTC = datetime(2020, 5, 21, 18, 28, 3, tzinfo=timezone.utc)   # mesmo epoch do SPICE
J2_ = 1.08262668e-3
RE_ = planet.radEquator

def _earth_rot_angle(epoch):
    """Ângulo (GMST) entre o eixo X inercial e o meridiano de Greenwich, no modelo do pck00010 (o mesmo do SPICE)."""
    d = ((epoch - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() + 69.184) / 86400.0
    W = 190.147 + 360.9856235 * d                     # W é medido a partir do nodo do equador (RA=90 deg)
    return np.radians((W + 90.0 - 0.641 * d / 36525.0) % 360.0)

def plan_pass(a, inc, lat, lon, epoch, orbit_target, raan_offset_deg=0.0):
    """Retorna (Omega0, u0) [rad] para que o satélite cruze a latitude da estação, em passagem ascendente,
    na longitude da estação, em t = orbit_target * período. Inclui a deriva secular de J2 (RAAN e arg. de latitude)."""
    n_ = np.sqrt(mu / a**3)
    k_ = 1.5 * J2_ * (RE_ / a)**2
    dOm = -k_ * n_ * np.cos(inc)                      # deriva de Omega
    du = n_ * (1.0 + k_ * (3.0 - 4.0 * np.sin(inc)**2))   # deriva de u = omega + f
    t_t = orbit_target * 2.0 * np.pi / n_
    u_t = np.arcsin(np.sin(lat) / np.sin(inc)) % (2.0 * np.pi)   # ramo ascendente
    dl = np.arctan2(np.cos(inc) * np.sin(u_t), np.cos(u_t))      # longitude inercial relativa ao nodo
    lam_t = _earth_rot_angle(epoch) + 7.2921159e-5 * t_t + lon   # longitude inercial da estação em t_t
    Om0 = (lam_t - dl - dOm * t_t + np.radians(raan_offset_deg)) % (2.0 * np.pi)
    u0 = (u_t - du * t_t) % (2.0 * np.pi)
    return Om0, u0

Om0_, u0_ = plan_pass(oe.a, oe.i, np.radians(lat_deg), np.radians(lon_deg), EPOCH_UTC,
                      PASS_TARGET_ORBIT, PASS_RAAN_OFFSET_DEG)
oe.Omega = Om0_
oe.f = (u0_ - oe.omega) % (2.0 * np.pi)               # u = omega + f (e ~ 0)
print(f"[ÓRBITA] Omega = {np.degrees(oe.Omega):.3f} deg | f = {np.degrees(oe.f):.3f} deg "
      f"(alvo: passagem em ~{PASS_TARGET_ORBIT} órbitas)")

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
mtbMomentumManagementObj.cGain = 0.0008
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
    [0.0,  1.0,  0.0],
    [0.0,  0.0,  1.0]])

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
stSunBAC.boreVec_B = [0., 0., 1.] 
stSunBAC.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
stSunBAC.celBodyInMsg.subscribeTo(sunStateMsg)
scSim.AddModelToTask(simTaskName, stSunBAC)

# terra
stEarthBAC = boreAngCalc.BoreAngCalc()
stEarthBAC.ModelTag = "starTrackerEarthBoresight"
stEarthBAC.boreVec_B = [0., 0., 1.] 
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

#bias máximo de 5° ~ TA=25°C
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

# Dipolo magnético residual parasita (CubeSat 3U típico: ~0.005 a 0.01 A*m^2 por eixo)
resDipoleCalc = ResidualMagneticDipole(m_res_B=[0.008, -0.006, 0.005])  # [A*m^2]
resDipoleCalc.magInMsg.subscribeTo(magModule.envOutMsgs[0])
resDipoleCalc.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
scSim.AddModelToTask(simTaskName, resDipoleCalc)

resDipoleEff = extForceTorque.ExtForceTorque()
resDipoleEff.ModelTag = "ResidualDipoleTorque"
resDipoleEff.cmdTorqueInMsg.subscribeTo(resDipoleCalc.torqueOutMsg)
scObject.addDynamicEffector(resDipoleEff)
scSim.AddModelToTask(simTaskName, resDipoleEff)

#groundlocation ~ senai cimatec
groundTarget = groundLocation.GroundLocation()
groundTarget.ModelTag = "groundTarget"
groundTarget.planetRadius = orbitalMotion.REQ_EARTH * 1000.
groundTarget.minimumElevation = np.radians(0.)

groundTarget.specifyLocation(
    lat_deg * macros.D2R,  # ou np.radians(lat_deg)
    lon_deg * macros.D2R,  # ou np.radians(lon_deg)
    alt_m) #  Sets location in latitude, longitude, altitude coordinates

# Conexões
groundTarget.addSpacecraftToModel(scObject.scStateOutMsg)
groundTarget.planetInMsg.subscribeTo(earthStateMsg)

scSim.AddModelToTask(simTaskName, groundTarget)

# groundComm: máscara operacional que marca a janela útil de comunicação
groundComm = groundLocation.GroundLocation()
groundComm.ModelTag = "groundComm"
groundComm.planetRadius = orbitalMotion.REQ_EARTH * 1000.
groundComm.minimumElevation = np.radians(30.)
groundComm.specifyLocation(lat_deg * macros.D2R, lon_deg * macros.D2R, alt_m)
groundComm.addSpacecraftToModel(scObject.scStateOutMsg)
groundComm.planetInMsg.subscribeTo(earthStateMsg)
scSim.AddModelToTask(simTaskName, groundComm)

# set the simulation time
n = np.sqrt(mu / oe.a / oe.a / oe.a)
P = 2.0 * np.pi / n

tempo_bdot = macros.sec2nano(N_ORBITS_BDOT * P)
totalSimTime = macros.sec2nano(N_ORBITS_TOTAL * P)
logDt = macros.sec2nano(LOG_PERIOD_S)

#definição do B-dot
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
scSim.AddModelToTask(fswTaskName, imuBridge)
biasCal = GyroBiasCalibrator(ImuSensor, tau=GYRO_BIAS_TAU_S)
scSim.AddModelToTask(fswTaskName, biasCal)

ukfObject.massPropsInMsg.subscribeTo(vcMsg)
ukfObject.gyrBuffInMsg.subscribeTo(imuBridge.accDataMsg)
ukfObject.rwParamsInMsg.subscribeTo(fswRwParamMsg)
ukfObject.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)

stBridge = StToUkfBridge(
    stSensorObj=ST, 
    dcm_CB_matrix=dcm_CB_matrix, 
    sunBacObj=stSunBAC, 
    earthBacObj=stEarthBAC, 
    fovThresholdDeg=40.0)
scSim.AddModelToTask(stTaskName, stBridge)

ukfObject.STDatasStruct.numST = 1
ukfObject.STDatasStruct.STMessages[0].stInMsg.subscribeTo(stBridge.stAttMsg)
R_st_case = np.diag([P_x, P_y, P_z])
R_st_body = dcm_CB_matrix.T @ R_st_case @ dcm_CB_matrix
R_st_mrp = R_st_body / 16.0  # Aproximação de pequenas rotações MRP

ukfObject.STDatasStruct.STMessages[0].noise = R_st_mrp.flatten().tolist()

ukfObject.dcm_BdyGyrpltf = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]   # plataforma do gyro = corpo

# Placeholders válidos (não-nulos - gemini) ~ basicamente um valor provisório apenas para poder instanciar o módulo
ukfObject.stateInit = [0.0] * 6
ukfObject.covarInit = np.diag([1e-4] * 3 + [1e-4] * 3).flatten().tolist()
ukfObject.qNoise = np.diag([1e-8] * 3 + [1e-6] * 3).flatten().tolist()

# Hot-start ~ condições inicias do UKF
P0_SIGMA = 1e-6                    # [MRP^2]  ST 
P0_OMEGA = np.radians(0.5) ** 2    # [(rad/s)^2] resíduo do gyro após remover o bias
Q_SIGMA = 1e-8
Q_OMEGA = 1e-6
P0_ukf = np.diag([P0_SIGMA] * 3 + [P0_OMEGA] * 3)
Q_ukf = np.diag([Q_SIGMA] * 3 + [Q_OMEGA] * 3)

scSim.AddModelToTask(nomTaskName, ukfObject, ModelPriority=100)

#definição dos sensores de atitude
sNavObject = simpleNav.SimpleNav()
sNavObject.ModelTag = "SimpleNavigation"
scSim.AddModelToTask(fswTaskName, sNavObject)

# setup guidance module
attGuidance = locationPointing.locationPointing()
attGuidance.ModelTag = "locationPointing"

# Estados reais do satélite
sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
attGuidance.pHat_B = [0.0, 0.0, -1.0]
attGuidance.useBoresightRateDamping = 1  # amortecimento de taxa angular na linha de visada

attGuidance.scTransInMsg.subscribeTo(sNavObject.transOutMsg)
attGuidance.scAttInMsg.subscribeTo(ukfObject.navStateOutMsg)
attGuidance.locationInMsg.subscribeTo(groundTarget.currentGroundStateOutMsg)
scSim.AddModelToTask(nomTaskName, attGuidance, ModelPriority=95)

#erro de atitude real
attGuidanceTruth = locationPointing.locationPointing()
attGuidanceTruth.ModelTag = "locationPointingTruth"
attGuidanceTruth.pHat_B = [0.0, 0.0, -1.0]
attGuidanceTruth.scTransInMsg.subscribeTo(sNavObject.transOutMsg)
attGuidanceTruth.scAttInMsg.subscribeTo(sNavObject.attOutMsg)
attGuidanceTruth.locationInMsg.subscribeTo(groundTarget.currentGroundStateOutMsg)
scSim.AddModelToTask(fswTaskName, attGuidanceTruth)

# setup the MRP Feedback control module
mrpControl = mrpFeedback.mrpFeedback()
mrpControl.ModelTag = "mrpFeedback"
scSim.AddModelToTask(nomTaskName, mrpControl, ModelPriority=80)

mrpControl.K = 0.008    # Proporcional
mrpControl.P = 0.033    # Derivativo 
mrpControl.Ki = 0.001    # Integral
mrpControl.integralLimit = 0.05

fswRwParamMsg = rwFactory.getConfigMessage()
vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

#links
mrpControl.guidInMsg.subscribeTo(attGuidance.attGuidOutMsg)
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

fswRWVoltage.VMin = 0.0 
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

# MTB começa comandado pelo B-dot
mtbEff.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
mtbEff.magInMsg.subscribeTo(magModule.envOutMsgs[0])

# RECORDERS
snLog = sNavObject.transOutMsg.recorder(logDt)
attNavLog = sNavObject.attOutMsg.recorder(logDt)
attErrorLog = attGuidanceTruth.attGuidOutMsg.recorder(logDt)
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
stLog = ST.sensorOutMsg.recorder(logDt)                         
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
resDipoleLog = resDipoleEff.logger(["torqueExternalPntB_B"], logDt)

for rec in (dragLog, srpLog, ggLog,resDipoleLog):
    scSim.AddModelToTask(simTaskName, rec)

# 2. DEFINIÇÃO DO ARQUIVO DE VISUALIZAÇÃO
viz = vizSupport.enableUnityVisualization(
    scSim, 
    simTaskName, 
    scObject,
    saveFile="sim3U_CubeSat.bin",
    rwEffectorList=rwStateEffector)

viz.epochInMsg.subscribeTo(gravFactory.epochMsg)
viz.settings.ambient = 0.8

vizSupport.createCustomModel(
    viz,
    modelPath=r"C:\Users\artur.azevedo\Downloads\cimatelite.glb",   
    simBodiesToModify=["cimatelite"],
    scale=[7.0, 7.0, 7.0],
    rotation=[0.0, np.pi / 2.0, 0.0])

vizSupport.addLocation(
    viz,
    stationName="SENAI CIMATEC",
    parentBodyName="earth",
    r_GP_P=np.array(groundComm.r_LP_P_Init).flatten().tolist(),
    fieldOfView=np.radians(120.0),  # cone correspondente a 30° de elevação mínima
    color=vizSupport.toRGBA255("cyan"),
    range=800.0 * 1000.0           # alcance visual do cone [m]
)

accessLog = groundTarget.accessOutMsgs[-1].recorder(logDt)
scSim.AddModelToTask(simTaskName, accessLog)

commAccessLog = groundComm.accessOutMsgs[-1].recorder(logDt)
scSim.AddModelToTask(simTaskName, commAccessLog)

scSim.InitializeSimulation()

# EXECUÇÃO: ETAPA 1 (B-dot) -> ETAPA 2 (alternância B-dot/UKF)
in_pointing_mode = False
mode_log = []                      # [(t_ns, 'AOS'|'LOS'), ...] -> máscara real do modo, para os gráficos
scSim.enableTask(bdotTaskName)
scSim.disableTask(nomTaskName)

# ETAPA 1: detumbling
scSim.ConfigureStopTime(tempo_bdot)
scSim.ExecuteSimulation()

bias_est = biasCal.bias.copy()
biasCal.frozen = True              # congela: durante o apontamento o gyro lê rotação real, não só bias
imuBridge.biasCorrection_B = bias_est
bias_est_deg = bias_est * macros.R2D
print(f"bias calibrado [deg/s]               : {np.round(bias_est_deg, 3)}")
print(f"bias verdadeiro (simulado) [deg/s]   : {np.round(gyro_bias_true_deg, 3)}")
print(f"erro de calibração [deg/s]           : {np.round(bias_est_deg - gyro_bias_true_deg, 3)}")

def will_reach_comm_elevation(min_comm_deg=30.0, horizon_s=420.0, step_s=10.0):
    """Propaga a órbita kepleriana e a rotação da Terra a partir do estado atual
    para verificar se a passagem atual atingirá min_comm_deg nos próximos horizon_s segundos."""
    sc_state = scObject.scStateOutMsg.read()
    r0 = np.array(sc_state.r_BN_N)
    v0 = np.array(sc_state.v_BN_N)
    oe_now = orbitalMotion.rv2elem(mu, r0, v0)
    n_now = np.sqrt(mu / (oe_now.a ** 3))

    gs_now = np.array(groundTarget.currentGroundStateOutMsg.read().r_LP_N)
    omega_earth = 7.2921159e-5  # [rad/s]

    for dt in np.arange(0.0, horizon_s + step_s, step_s):
        oe_fut = orbitalMotion.ClassicElements()
        oe_fut.a = oe_now.a
        oe_fut.e = oe_now.e
        oe_fut.i = oe_now.i
        oe_fut.Omega = oe_now.Omega
        oe_fut.omega = oe_now.omega
        oe_fut.f = oe_now.f + n_now * dt
        r_sat_fut, _ = orbitalMotion.elem2rv(mu, oe_fut)

        ang = omega_earth * dt
        c, s = np.cos(ang), np.sin(ang)
        r_gs_fut = np.array([
            c * gs_now[0] - s * gs_now[1],
            s * gs_now[0] + c * gs_now[1],
            gs_now[2]
        ])

        los_vec = r_sat_fut - r_gs_fut
        zenith_hat = r_gs_fut / np.linalg.norm(r_gs_fut)
        sin_el = np.dot(los_vec, zenith_hat) / np.linalg.norm(los_vec)
        el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

        if el_deg >= min_comm_deg:
            return True

    return False


# ETAPA 2
check_step = macros.sec2nano(15.0)  # resolução de detecção AOS/LOS
current_time = tempo_bdot
next_print = 5.0
pass_evaluated = False             # evita reavaliar a mesma passagem de 0° a cada passo
pass_valid_for_comm = False        # True somente se a passagem atingir >= 30°

while current_time < totalSimTime:
    next_time = min(current_time + check_step, totalSimTime)
    scSim.ConfigureStopTime(next_time)
    scSim.ExecuteSimulation()
    current_time = next_time

    has_access = int(groundTarget.accessOutMsgs[-1].read().hasAccess)
    in_eclipse = eclipseObject.eclipseOutMsgs[0].read().shadowFactor < 0.5
    t_orb = current_time * macros.NANO2SEC / P

    if has_access == 1:
        # Avalia uma única vez ao cruzar o horizonte (0°) se a passagem vai cruzar >= 30°
        if not pass_evaluated:
            pass_valid_for_comm = will_reach_comm_elevation(min_comm_deg=30.0)
            pass_evaluated = True
            if not pass_valid_for_comm:
                print(f"[INFO] Passagem em t = {t_orb:.3f} órbitas descartada (elevação máx < 30°)")

        # Liga o UKF apenas se: atingirá 30°, está em eclipse (noite) e o ST não está cego
        if pass_valid_for_comm and in_eclipse and not in_pointing_mode:
            if st_is_blind():
                continue               # sem ST válido não há hot-start confiável: espera sair do cone
            print(f"[EVENTO] AOS (pre-slew 0° em eclipse) em t = {t_orb:.3f} órbitas -> UKF + LocationPointing")
            enter_pointing_mode(current_time)
            in_pointing_mode = True
            mode_log.append((current_time, 'AOS'))

        # Se sair do eclipse no meio da passagem, desliga o apontamento fino
        elif (not in_eclipse) and in_pointing_mode:
            print(f"[EVENTO] Saída do eclipse em t = {t_orb:.3f} órbitas -> B-dot")
            enter_bdot_mode(current_time)
            in_pointing_mode = False
            mode_log.append((current_time, 'LOS'))

    else:
        # Saiu do horizonte de 0°: reseta flags para a próxima passagem
        pass_evaluated = False
        pass_valid_for_comm = False

        if in_pointing_mode:
            print(f"[EVENTO] LOS (0°) em t = {t_orb:.3f} órbitas -> B-dot")
            enter_bdot_mode(current_time)
            in_pointing_mode = False
            mode_log.append((current_time, 'LOS'))

    if t_orb >= next_print:
        print(f"   ... t = {t_orb:.2f} órbitas")
        next_print += 0.5

if in_pointing_mode:
    mode_log.append((current_time, 'LOS'))   # fecha janela aberta no fim da simulação

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
                  ("attNavLog", attNavLog), ("imuLog", imuLog), ("stLog", stLog), ("tamLog", tamLog),
                  ("accessLog", accessLog)]:
    if len(rec.times()) != len(snLog.times()):
        print(f"[AVISO] {nome}: {len(rec.times())} pontos != {len(snLog.times())} (snLog). "
              "Verifique se está em tarefa sempre ativa.")

# 1. Atitude real
sigma_BN_real = np.array(scLog.sigma_BN)
omega_real_rad = np.array(scLog.omega_BN_B)
quat_BN_real = np.array([rbk.MRP2EP(s_) for s_ in sigma_BN_real])
omega_BN_real_deg = omega_real_rad * macros.R2D
omega_true_norm = np.linalg.norm(omega_BN_real_deg, axis=1)

# 2. Erro de atitude (verdade em relação à Estação de Solo)
attErrSigma_global = attErrorLog.sigma_BR
attErrOmega_global = attErrorLog.omega_BR_B
quatErr_global = np.array([rbk.MRP2EP(s_) for s_ in attErrSigma_global])
att_err_target_deg = mrp_angle_deg(attErrSigma_global)

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

# ==============================================================================
# MÁSCARAS DE REGIME (DETUMBLING vs. VISADA OPERACIONAL >= 4 ÓRBITAS)
# ==============================================================================
has_access_sim = np.array(accessLog.hasAccess, dtype=int) == 1     # visibilidade geométrica (referência)
# Modo REALMENTE ativo (UKF + RW): intervalos [AOS, LOS) registrados no laço de execução
t_ns_sim = np.array(snLog.times())
mode_pointing = np.zeros(len(t_ns_sim), dtype=bool)
for (t_a, tag_a), (t_l, tag_l) in zip(mode_log[0::2], mode_log[1::2]):
    mode_pointing |= (t_ns_sim >= t_a) & (t_ns_sim < t_l)

m_b1_sim = timeOrbits_sim < (N_ORBITS_BDOT - 1e-3)     # Detumbling inicial (0 -> 4 órbitas)
m_b2_sim = timeOrbits_sim >= N_ORBITS_BDOT              # Regime alternado (>= 4 órbitas)
m_pointing_active = m_b2_sim & mode_pointing            # UKF + LocationPointing + RWs ativos (AOS)
m_bdot_wait = m_b2_sim & (~mode_pointing)               # B-dot fora de visada (LOS)

# Erros do UKF calculados apenas quando o UKF está ativo (dentro de visada após 4ª órbita)
mask_ukf = m_pointing_active
if np.any(mask_ukf):
    att_err_ukf_deg = mrp_angle_deg(np.array([rbk.subMRP(a, b) for a, b in
                                              zip(sigma_BN_ukf[mask_ukf], sigma_BN_real[mask_ukf])]))
    omega_err_ukf_deg = omega_ukf_deg[mask_ukf] - omega_BN_real_deg[mask_ukf]
else:
    att_err_ukf_deg = np.array([])
    omega_err_ukf_deg = np.empty((0, 3))

has_comm_sim = np.array(commAccessLog.hasAccess, dtype=int) == 1
m_comm_active = m_b2_sim & has_comm_sim & mode_pointing   # janela real de comunicação (>= 30 deg) com UKF ativo

# ==============================================================================
# IMPRESSÃO DO DIAGNÓSTICO DE DESEMPENHO NO TERMINAL
# ==============================================================================
print("\n===================== DIAGNÓSTICO =====================")
print("ETAPA 1: Detumbling Exclusivo (0 -> 4 órbitas)")
print(f"   ||omega|| inicial / final (real)    : {omega_true_norm[0]:.2f} / {omega_true_norm[m_b1_sim][-1]:.3f} deg/s")

print(f"   janelas AOS/LOS executadas          : {len(mode_log)//2}")
for k_, ((t_a, _), (t_l, _)) in enumerate(zip(mode_log[0::2], mode_log[1::2]), 1):
    print(f"     passagem {k_}: {t_a*macros.NANO2SEC/P:.3f} -> {t_l*macros.NANO2SEC/P:.3f} órbitas ({(t_l-t_a)*macros.NANO2SEC:.0f} s)")
print(f"ETAPA 2: Regime Alternado (>= 4 órbitas) | Pontos em visada: {np.sum(m_pointing_active)}")
if np.any(m_pointing_active):
    print(f"   erro apontamento mínimo (em visada) : {np.min(att_err_target_deg[m_pointing_active]):.3f} deg")
    print(f"   max |Omega_RW| (em visada)          : {np.max(np.abs(dataOmegaRW[m_pointing_active])):.0f} RPM (saturação = 15000)")
    print(f"   max |V_RW| (em visada)              : {np.max(np.abs(dataVoltRW[m_pointing_active])):.2f} V (limite {V_MAX})")
    if att_err_ukf_deg.size:
        print(f"   erro de atitude do UKF (RMS visada) : {np.sqrt(np.mean(att_err_ukf_deg**2)):.4f} deg")
        print(f"   erro de omega do UKF (RMS visada)   : {np.sqrt(np.mean(omega_err_ukf_deg**2, axis=0)).round(4)} deg/s")
    print(f"   ||omega|| real médio (em visada)    : {np.mean(omega_true_norm[m_pointing_active]):.3f} deg/s")
else:
    print("   [AVISO] Nenhuma janela de visada registrada após a 4ª órbita.")
if np.any(m_bdot_wait):
    print(f"   ||omega|| real médio (espera B-dot) : {np.mean(omega_true_norm[m_bdot_wait]):.3f} deg/s")
print("========================================================\n")

# ==============================================================================
# 6. PLOTAGEM ESTRUTURADA E DIAGNÓSTICO DO UKF
# ==============================================================================
plt.close("all")

m_imu = timeOrbits_imu >= N_ORBITS_BDOT
m_sim = timeOrbits_sim >= N_ORBITS_BDOT
m_ukf = timeOrbits_ukf_log >= N_ORBITS_BDOT
t_u = timeOrbits_ukf_log[mask_ukf]

def shade_access(ax, mask, t_vec):
    """Destaca em verde claro as janelas de visada (AOS) no fundo do gráfico."""
    if np.any(mask):
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t_vec, ymin, ymax, where=mask, color='tab:green', alpha=0.14, label='Janela de Visada (AOS)')
        ax.set_ylim(ymin, ymax)

# ------------------------------------------------------------------------------
# FIGURA 1: Erro de Atitude (Quatérnio de Erro em Relação à Estação) - verdade
# ------------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(timeOrbits_sim, quatErr_global[:, 0], label=r'$q_0$ (Escalar)', color='black', linewidth=1.2)
ax1.plot(timeOrbits_sim, quatErr_global[:, 1], label=r'$q_1$ (Eixo $x$)', color='tab:red', linewidth=1.0)
ax1.plot(timeOrbits_sim, quatErr_global[:, 2], label=r'$q_2$ (Eixo $y$)', color='tab:green', linewidth=1.0)
ax1.plot(timeOrbits_sim, quatErr_global[:, 3], label=r'$q_3$ (Eixo $z$)', color='tab:blue', linewidth=1.0)
ax1.axvline(N_ORBITS_BDOT, color='gray', linestyle=':', label='Fim Detumbling')
shade_access(ax1, m_pointing_active, timeOrbits_sim)
ax1.set_title(r"Evolução do Quatérnio de Erro em Relação à Estação ($q_{B/R}$) - verdade")
ax1.set_xlabel("Tempo [Órbitas]")
ax1.set_ylabel("Componentes $q$")
ax1.legend(loc="upper right", ncol=2)
ax1.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 1B: Erro de Atitude Real em Relação à Estação (Escala Log, Órbita >= 4)
# ------------------------------------------------------------------------------
q_err_b2 = quatErr_global[m_sim]

fig1b, ax1b = plt.subplots(figsize=(9, 4))
ax1b.semilogy(timeOrbits_sim[m_sim], np.maximum(np.abs(1.0 - np.abs(q_err_b2[:, 0])), 1e-8), color='black', linewidth=1.2, label=r'$|1 - |q_0||$ (Escalar)')
ax1b.semilogy(timeOrbits_sim[m_sim], np.maximum(np.abs(q_err_b2[:, 1]), 1e-8), color='tab:red', linewidth=1.0, label=r'$|q_1|$ (Eixo $x$)')
ax1b.semilogy(timeOrbits_sim[m_sim], np.maximum(np.abs(q_err_b2[:, 2]), 1e-8), color='tab:green', linewidth=1.0, label=r'$|q_2|$ (Eixo $y$)')
ax1b.semilogy(timeOrbits_sim[m_sim], np.maximum(np.abs(q_err_b2[:, 3]), 1e-8), color='tab:blue', linewidth=1.0, label=r'$|q_3|$ (Eixo $z$)')
shade_access(ax1b, m_pointing_active[m_sim], timeOrbits_sim[m_sim])
ax1b.set_title(r"Erro de Atitude Real em Relação à Estação ($q_{B/R}$) a partir da 4ª Órbita (Escala Log)")
ax1b.set_xlabel("Tempo [Órbitas]")
ax1b.set_ylabel("Componentes $q$")
ax1b.grid(True, which="both", linestyle="--", alpha=0.5)
ax1b.legend(loc="upper right", ncol=2)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 1C: Precisão de Apontamento Real Apenas na Janela de Comunicação (>= 30°)
# ------------------------------------------------------------------------------
fig1c, axes1c = plt.subplots(6, 1, figsize=(10.5, 15.0), sharex=True)
ax1c_q, ax1c_ang, ax1c_xyz_log, ax1c_x_lin, ax1c_y_lin, ax1c_z_lin = axes1c

comm_b2 = m_comm_active[m_sim]   # somente janela útil de comunicação (>= 30 deg)
t_b2 = timeOrbits_sim[m_sim]

q0_err_comm = np.where(comm_b2, np.maximum(np.abs(1.0 - np.abs(q_err_b2[:, 0])), 1e-8), np.nan)
q1_err_comm = np.where(comm_b2, np.maximum(np.abs(q_err_b2[:, 1]), 1e-8), np.nan)
q2_err_comm = np.where(comm_b2, np.maximum(np.abs(q_err_b2[:, 2]), 1e-8), np.nan)
q3_err_comm = np.where(comm_b2, np.maximum(np.abs(q_err_b2[:, 3]), 1e-8), np.nan)
ang_err_comm = np.where(comm_b2, np.maximum(att_err_target_deg[m_sim], 1e-5), np.nan)

# Erros angulares com sinal por eixo [deg] (escala linear)
theta_x_signed = 2.0 * np.degrees(np.arcsin(np.clip(q_err_b2[:, 1], -1.0, 1.0)))
theta_y_signed = 2.0 * np.degrees(np.arcsin(np.clip(q_err_b2[:, 2], -1.0, 1.0)))
theta_z_signed = 2.0 * np.degrees(np.arcsin(np.clip(q_err_b2[:, 3], -1.0, 1.0)))

theta_x_lin_comm = np.where(comm_b2, theta_x_signed, np.nan)
theta_y_lin_comm = np.where(comm_b2, theta_y_signed, np.nan)
theta_z_lin_comm = np.where(comm_b2, theta_z_signed, np.nan)

# Módulo dos erros angulares por eixo [deg] com piso numérico (escala log)
theta_x_log_comm = np.where(comm_b2, np.maximum(np.abs(theta_x_signed), 1e-6), np.nan)
theta_y_log_comm = np.where(comm_b2, np.maximum(np.abs(theta_y_signed), 1e-6), np.nan)
theta_z_log_comm = np.where(comm_b2, np.maximum(np.abs(theta_z_signed), 1e-6), np.nan)

# Subplot 1: Componentes do quatérnio de erro (Log)
ax1c_q.semilogy(t_b2, q0_err_comm, color='black', linewidth=1.3, label=r'$|1 - |q_0||$ (Escalar)')
ax1c_q.semilogy(t_b2, q1_err_comm, color='tab:red', linewidth=1.1, label=r'$|q_1|$ (Eixo $x$)')
ax1c_q.semilogy(t_b2, q2_err_comm, color='tab:green', linewidth=1.1, label=r'$|q_2|$ (Eixo $y$)')
ax1c_q.semilogy(t_b2, q3_err_comm, color='tab:blue', linewidth=1.1, label=r'$|q_3|$ (Eixo $z$)')
shade_access(ax1c_q, comm_b2, t_b2)
ax1c_q.set_ylabel("Comp. $q_{B/R}$")
ax1c_q.set_title(r"Precisão de Apontamento Real Durante a Janela de Comunicação ($\geq 30^\circ$)")
ax1c_q.grid(True, which="both", linestyle="--", alpha=0.5)
ax1c_q.legend(loc="upper right", ncol=2, fontsize=8.5)

# Subplot 2: Erro angular total [deg] (Log)
ax1c_ang.semilogy(t_b2, ang_err_comm, color='purple', linewidth=1.5, label=r'Erro Angular Total $\theta_{B/R}$ [deg]')
shade_access(ax1c_ang, comm_b2, t_b2)
ax1c_ang.set_ylabel("Total [deg]")
ax1c_ang.grid(True, which="both", linestyle="--", alpha=0.5)
ax1c_ang.legend(loc="upper right", fontsize=8.5)

# Subplot 3: Erro angular por eixo X, Y, Z [deg] (Log)
ax1c_xyz_log.semilogy(t_b2, theta_x_log_comm, color='tab:red', linewidth=1.2, label=r'$|\theta_x|$ (Eixo $x$)')
ax1c_xyz_log.semilogy(t_b2, theta_y_log_comm, color='tab:green', linewidth=1.2, label=r'$|\theta_y|$ (Eixo $y$)')
ax1c_xyz_log.semilogy(t_b2, theta_z_log_comm, color='tab:blue', linewidth=1.2, label=r'$|\theta_z|$ (Eixo $z$)')
shade_access(ax1c_xyz_log, comm_b2, t_b2)
ax1c_xyz_log.set_ylabel("Log Eixos [deg]")
ax1c_xyz_log.grid(True, which="both", linestyle="--", alpha=0.5)
ax1c_xyz_log.legend(loc="upper right", ncol=2, fontsize=8.5)

# Subplot 4: Erro Eixo X [deg] (Escala Linear)
ax1c_x_lin.plot(t_b2, theta_x_lin_comm, color='tab:red', linewidth=1.3, label=r'$\theta_x$ (Eixo $x$ - Linear)')
ax1c_x_lin.axhline(0.0, color='gray', linestyle=':', linewidth=0.9)
shade_access(ax1c_x_lin, comm_b2, t_b2)
ax1c_x_lin.set_ylabel(r"$\theta_x$ [deg]")
ax1c_x_lin.grid(True, linestyle="--", alpha=0.5)
ax1c_x_lin.legend(loc="upper right", fontsize=8.5)

# Subplot 5: Erro Eixo Y [deg] (Escala Linear)
ax1c_y_lin.plot(t_b2, theta_y_lin_comm, color='tab:green', linewidth=1.3, label=r'$\theta_y$ (Eixo $y$ - Linear)')
ax1c_y_lin.axhline(0.0, color='gray', linestyle=':', linewidth=0.9)
shade_access(ax1c_y_lin, comm_b2, t_b2)
ax1c_y_lin.set_ylabel(r"$\theta_y$ [deg]")
ax1c_y_lin.grid(True, linestyle="--", alpha=0.5)
ax1c_y_lin.legend(loc="upper right", fontsize=8.5)

# Subplot 6: Erro Eixo Z [deg] (Escala Linear)
ax1c_z_lin.plot(t_b2, theta_z_lin_comm, color='tab:blue', linewidth=1.3, label=r'$\theta_z$ (Eixo $z$ - Linear)')
ax1c_z_lin.axhline(0.0, color='gray', linestyle=':', linewidth=0.9)
shade_access(ax1c_z_lin, comm_b2, t_b2)
ax1c_z_lin.set_xlabel("Tempo [Órbitas]")
ax1c_z_lin.set_ylabel(r"$\theta_z$ [deg]")
ax1c_z_lin.grid(True, linestyle="--", alpha=0.5)
ax1c_z_lin.legend(loc="upper right", fontsize=8.5)

# Ajusta o zoom do eixo X exatamente para o intervalo da(s) janela(s) de comunicação
if np.any(comm_b2):
    t_comm_valid = t_b2[comm_b2]
    margin = 0.05 * max(t_comm_valid[-1] - t_comm_valid[0], 1e-4)
    ax1c_z_lin.set_xlim(t_comm_valid[0] - margin, t_comm_valid[-1] + margin)

plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 2: Erro de Velocidade Angular - verdade
# ------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 4))
omega_deg = attErrOmega_global * macros.R2D
ax2.plot(timeOrbits_sim, omega_deg[:, 0], label=r'$\omega_{x}$', color='tab:red')
ax2.plot(timeOrbits_sim, omega_deg[:, 1], label=r'$\omega_{y}$', color='tab:green')
ax2.plot(timeOrbits_sim, omega_deg[:, 2], label=r'$\omega_{z}$', color='tab:blue')
ax2.axvline(N_ORBITS_BDOT, color='gray', linestyle=':', label='Fim Detumbling')
shade_access(ax2, m_pointing_active, timeOrbits_sim)
ax2.set_title(r"Erro de Velocidade Angular ($\omega_{B/R}$) - verdade")
ax2.set_xlabel("Tempo [Órbitas]")
ax2.set_ylabel("Velocidade Angular [deg/s]")
ax2.legend(loc="upper right", ncol=2)
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
shade_access(ax3, m_pointing_active, timeOrbits_sim)
ax3.set_title("Velocidade Angular das Rodas de Reação Hyperion RW210")
ax3.set_xlabel("Tempo [Órbitas]")
ax3.set_ylabel("Velocidade [RPM]")
ax3.legend(loc="upper right", ncol=2)
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
shade_access(ax4, m_pointing_active, timeOrbits_sim)
ax4.set_title("Torque Magnético Gerado pelo MTB Hyperion HT-MTQ200")
ax4.set_xlabel("Tempo [Órbitas]")
ax4.set_ylabel(r"Torque [$\mu\text{N}\cdot\text{m}$]")
ax4.legend(loc="upper right", ncol=2)
ax4.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 4B: Torques de Perturbação Ambiental em Escala Logarítmica
# ------------------------------------------------------------------------------
fig4b, axes_dist = plt.subplots(4, 1, figsize=(10.5, 11.5), sharex=True)
ax_mag_d, ax_gg_d, ax_srp_d, ax_drag_d = axes_dist

# Extrai vetores [N*m] e converte para [uN*m]
tau_res_uNm  = np.array(resDipoleLog.torqueExternalPntB_B) * 1e6
tau_gg_uNm   = np.array(ggLog.torqueExternalPntB_B) * 1e6
tau_srp_uNm  = np.array(srpLog.torqueExternalPntB_B) * 1e6
tau_drag_uNm = np.array(dragLog.torqueExternalPntB_B) * 1e6

dist_configs = [
    (ax_mag_d,  tau_res_uNm,  "Dipolo Magnético Residual ($m_{res} \\times B$)"),
    (ax_gg_d,   tau_gg_uNm,   "Gradiente de Gravidade (GG)"),
    (ax_srp_d,  tau_srp_uNm,  "Pressão de Radiação Solar (SRP)"),
    (ax_drag_d, tau_drag_uNm, "Arrasto Aerodinâmico (Aerodrag)")
]

floor_uNm = 1e-6  # piso numérico [uN*m] para evitar log(0)

for ax_d, tau_data, title_str in dist_configs:
    tau_norm = np.linalg.norm(tau_data, axis=1)

    ax_d.semilogy(timeOrbits_sim, np.maximum(np.abs(tau_data[:, 0]), floor_uNm),
                  label=r'$|\tau_x|$', color='tab:red', linewidth=1.0, alpha=0.85)
    ax_d.semilogy(timeOrbits_sim, np.maximum(np.abs(tau_data[:, 1]), floor_uNm),
                  label=r'$|\tau_y|$', color='tab:green', linewidth=1.0, alpha=0.85)
    ax_d.semilogy(timeOrbits_sim, np.maximum(np.abs(tau_data[:, 2]), floor_uNm),
                  label=r'$|\tau_z|$', color='tab:blue', linewidth=1.0, alpha=0.85)
    ax_d.semilogy(timeOrbits_sim, np.maximum(tau_norm, floor_uNm),
                  label=r'$\|\tau\|$', color='black', linewidth=1.4)

    ax_d.axvline(N_ORBITS_BDOT, color='gray', linestyle=':', linewidth=1.0)
    shade_access(ax_d, m_pointing_active, timeOrbits_sim)
    ax_d.set_ylabel(r"Torque [$\mu\text{N}\cdot\text{m}$]")
    ax_d.set_title(title_str, fontsize=10.5)
    ax_d.grid(True, which="both", linestyle="--", alpha=0.5)

axes_dist[0].legend(loc="upper right", ncol=5, fontsize=8.5)
axes_dist[-1].set_xlabel("Tempo [Órbitas]")
fig4b.suptitle("Comparação dos Torques de Perturbação Ambiental (Escala Log)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 7: Regime Alternado (Órbitas >= 4): Velocidade Angular em ESCALA LOG
# ------------------------------------------------------------------------------
fig7, ax7 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
labels_w = [r'$|\omega_x|$ [deg/s]', r'$|\omega_y|$ [deg/s]', r'$|\omega_z|$ [deg/s]']

# Mascarar com NaN a estimativa do UKF quando ele estiver desligado (fora de visada)
omega_ukf_plot = np.where(m_pointing_active[:, None], omega_ukf_deg, np.nan)

for i in range(3):
    ax7[i].plot(timeOrbits_sim[m_sim], np.abs(omega_BN_real_deg[m_sim, i]), label=r'Real ($|\omega_{B/N}|$)', color='black', linewidth=1.2)
    ax7[i].plot(timeOrbits_imu[m_imu], np.abs(omega_BN_measured_deg[m_imu, i]), label=r'IMU (com bias)', color='lightcoral' if i==0 else ('lightgreen' if i==1 else 'skyblue'), linestyle='--', alpha=0.7, linewidth=1.0)
    ax7[i].plot(timeOrbits_ukf_log[m_ukf], np.abs(omega_ukf_plot[m_ukf, i]), label=r'Estimado UKF (AOS)', color='darkorange', linewidth=1.2)
    ax7[i].set_yscale('log')
    ax7[i].set_ylabel(labels_w[i])
    shade_access(ax7[i], m_pointing_active[m_sim], timeOrbits_sim[m_sim])
    ax7[i].grid(True, which="both", linestyle="--", alpha=0.5)
    if i == 0:
        ax7[i].legend(loc="upper right", ncol=2)

ax7[2].set_xlabel("Tempo [Órbitas]")
fig7.suptitle("Regime Alternado (Órbitas >= 4): Módulo da Velocidade Angular (Escala Logarítmica)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 8: Estimativa de Atitude: Real (Pontilhada) vs. UKF nas Janelas AOS
# ------------------------------------------------------------------------------
fig8, ax8 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
labels_q = [r'$q_0$ (Escalar)', r'$q_1$ (Eixo $x$)', r'$q_2$ (Eixo $y$)', r'$q_3$ (Eixo $z$)']
colors_q = ['black', 'tab:red', 'tab:green', 'tab:blue']

quat_ukf_plot = np.where(m_pointing_active[:, None], quat_BN_ukf, np.nan)

for i in range(4):
    ax8[i].plot(timeOrbits_sim[m_sim], quat_BN_real[m_sim, i], label='Atitude Real', color='black', linestyle=':', linewidth=2.0)
    ax8[i].plot(timeOrbits_ukf_log[m_ukf], quat_ukf_plot[m_ukf, i], label='Atitude Estimada (UKF - AOS)', color=colors_q[i], linewidth=1.4, alpha=0.9)
    ax8[i].set_ylabel(labels_q[i])
    shade_access(ax8[i], m_pointing_active[m_sim], timeOrbits_sim[m_sim])
    ax8[i].grid(True, linestyle="--", alpha=0.5)
    if i == 0:
        ax8[i].legend(loc="upper right", ncol=3)

ax8[3].set_xlabel("Tempo [Órbitas]")
fig8.suptitle("Estimativa de Atitude: Real (Pontilhada) vs. UKF nas Janelas de Visada (Contínua)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 9: Erro de Atitude Real vs. Estimado pelo UKF (Somente Janelas AOS)
# ------------------------------------------------------------------------------
fig9, ax9 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
dq_ukf_real = np.zeros_like(quat_BN_ukf[m_ukf])

for k_ in range(np.sum(m_ukf)):
    dq_ukf_real[k_] = rbk.subEP(quat_BN_ukf[m_ukf][k_], quat_BN_real[m_sim][k_])

labels_q_log = [
    r'$|1 - |\delta q_0||$ (Escalar)',
    r'$|\delta q_1|$ (Eixo $x$)',
    r'$|\delta q_2|$ (Eixo $y$)',
    r'$|\delta q_3|$ (Eixo $z$)'
]

active_submask = m_pointing_active[m_ukf]
for i in range(4):
    if i == 0:
        val_plot = np.abs(1.0 - np.abs(dq_ukf_real[:, 0]))
    else:
        val_plot = np.abs(dq_ukf_real[:, i])
    val_plot = np.where(active_submask, np.maximum(val_plot, 1e-8), np.nan)

    ax9[i].semilogy(timeOrbits_ukf_log[m_ukf], val_plot, color=colors_q[i], linewidth=1.2, label=labels_q_log[i])
    ax9[i].set_ylabel(labels_q_log[i])
    shade_access(ax9[i], active_submask, timeOrbits_ukf_log[m_ukf])
    ax9[i].grid(True, which="both", linestyle="--", alpha=0.5)
    ax9[i].legend(loc="upper right")

ax9[3].set_xlabel("Tempo [Órbitas]")
fig9.suptitle("Erro de Atitude entre Real e Estimado pelo UKF nas Janelas AOS (Escala Logarítmica)", fontsize=12)
plt.tight_layout()

# ------------------------------------------------------------------------------
# FIGURA 12: Visibilidade e Ângulos de Exclusão do Star Tracker (Cegueira)
# ------------------------------------------------------------------------------
fig12, (ax12_1, ax12_2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

m_st_b2 = timeOrbits_st >= N_ORBITS_BDOT
t_st_b2 = timeOrbits_st[m_st_b2]

ax12_1.plot(t_st_b2, sun_angles_interp[m_st_b2], label='Ângulo do Sol (Boresight)', color='tab:orange', linewidth=1.2)
ax12_1.plot(t_st_b2, earth_angles_interp[m_st_b2], label='Ângulo da Terra (Boresight)', color='tab:blue', linewidth=1.2)
ax12_1.axhline(fov_threshold_deg, color='red', linestyle='--', linewidth=1.0, label=f'Limiar de Cegueira ({fov_threshold_deg}°)')
shade_access(ax12_1, m_pointing_active[m_sim], timeOrbits_sim[m_sim])
ax12_1.set_ylabel("Ângulo de Apontamento [deg]")
ax12_1.set_title("Indicadores de Visada e Cegueira do Star Tracker (ST-200)", fontsize=12)
ax12_1.grid(True, linestyle="--", alpha=0.5)
ax12_1.legend(loc="upper right")

st_blind_flag = st_blind_mask[m_st_b2].astype(int)

ax12_2.step(t_st_b2, st_blind_flag, color='tab:red', where='post', linewidth=1.5, label='Status de Cegueira (1 = Ocluso / 0 = Operacional)')
ax12_2.fill_between(t_st_b2, 0, st_blind_flag, color='tab:red', alpha=0.2, step='post')
ax12_2.set_yticks([0, 1])
ax12_2.set_yticklabels(['Válido (0)', 'Ocluso (1)'])
ax12_2.set_xlabel("Tempo [Órbitas]")
ax12_2.set_ylabel("Status do ST")
ax12_2.grid(True, linestyle="--", alpha=0.5)
ax12_2.legend(loc="upper right")

plt.tight_layout()

plt.show()