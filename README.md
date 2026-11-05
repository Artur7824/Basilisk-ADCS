# 🛰️ CubeSat 3U ADCS Simulation - Basilisk

Este repositório contém o processo de desenvolvimento de uma simulação do sistema de Determinação e Controle de Atitude (ADCS) para um **CubeSat 3U**, desenvolvida utilizando o framework **Basilisk** em ambiente **Google Colab**.

O projeto final simula a transição operacional da espaçonave em dois estágios principais de missão:
1. **Estágio 1 — Detumbling:** Atuação via Magnetorquers (**Hyperion HT-MTQ200**) utilizando a lei de controle **B-dot**.
2. **Estágio 2 — Estabilização Nominal (Apontamento Nadir):** Atuação via 3x _Reaction Wheels_ (**Hyperion RW210**), com estimação de atitude via **_Unscented Kalman Filter_ (UKF)** integrando dados do  Giroscópio (**MPU-9250**) e _Star Tracker_ (**ST-200**).

---

## 📁 Estrutura do Repositório

```text
Basilisk - ADCS/
├── vizard/                  # Modelo .glb para visualização 3D no Vizard
├── references/              # Documentação técnica e artigos  
├── examples/                # Scripts e cenários de referência do próprio Basilisk
├── testes/                  # Etapas graduais, simulações intermediárias e testes do Colab
├── README.md                # Documentação principal do projeto
└── requirements.txt         # Dependências das bibliotecas Python
