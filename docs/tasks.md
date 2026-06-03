# 📋 Quadro de Tarefas: Bot Detection System (Twitter Simulation)

**Status do Projeto:** Ingestão, Feature Store, Inferência, MLOps e Resiliência Concluídos (Todos os Marcos 1-6 100% Verdes)
**Metodologia:** Kanban / Milestones
**Base Científica:** CALEB (arXiv:2205.15707), Cresci et al. (2017), Twibot-20.

---

## 🎯 Prioridades Globais
1. [x] Infraestrutura e Persistência de Estado (Containers).
2. [x] Geração de Dados (Tráfego de simulação estilo Twitter com *Dual-Labeling*).
3. [x] Processamento de Stream e Feature-Engineering real (Redis & Neo4j).
4. [x] Inferência ML Híbrida (FastAPI GraphSAGE + Bi-GRU com pesos *Twibot*).
5. [x] MLOps, Resiliência e Explicabilidade.

---

## 🗺️ Marcos e Tarefas (Backlog & Status)

### Marco 1: Infraestrutura Base (A Fundação)
*Objetivo: Ter os serviços de armazenamento de pé e a comunicar entre si.*

- [x] **Task 1.1:** Escrever o `docker-compose.yml` utilizando os padrões atuais de mercado (Kafka em modo KRaft, Redis, Neo4j). **[CONCLUÍDO]**
- [x] **Task 1.2:** Criar script de setup `init_infra.py` para automatizar a criação de tópicos no Kafka e estabelecer índices/constraints de performance no Neo4j. **[CONCLUÍDO]**

### Marco 2: Simulador Adversarial (A Fonte de Dados)
*Objetivo: Gerar tráfego realista estilo Twitter usando IA local e injetar no Kafka.*

- [x] **Task 2.1:** Configurar cliente local do Ollama no Python (Phi-3 ou Llama-3). **[CONCLUÍDO]**
- [x] **Task 2.2:** Desenvolver as classes `NormalUserAgent` e `BotAgent`. O gerador deve produzir textos curtos (estilo *tweet*), incluindo menções (`@`) e *hashtags* (`#`). **[CONCLUÍDO]**
- [x] **Task 2.3:** Escrever o loop de simulação aplicando a estratégia de **Dual-Labeling**: o JSON enviado para o Kafka deve conter a ação do utilizador E uma *flag* oculta `"true_label"` (que o modelo ML nunca verá) para auditoria final. **[CONCLUÍDO]**

### Marco 3: Processador de Stream (O Estado)
*Objetivo: Consumir os dados em tempo real e desenhar o estado temporal e espacial do utilizador.*

- [x] **Task 3.1:** Desenvolver o Consumer Kafka de alta performance utilizando `confluent-kafka`. **[CONCLUÍDO]**
- [x] **Task 3.2:** Implementar a lógica temporal no Redis (listas circulares com as últimas 10 ações e preenchimento de padding). **[CONCLUÍDO]**
- [x] **Task 3.3:** Implementar a topologia de rede no Neo4j via Cypher (criar nós e arestas direcionadas `FOLLOWS`, `RETWEETS`, `MENTIONS`). **[CONCLUÍDO]**

### Marco 4: Motor de Inferência (A Decisão)
*Objetivo: Avaliar o risco de cada ação usando Transfer Learning.*

- [x] **Task 4.1:** Criar a estrutura base da API de inferência com FastAPI (`POST /predict`). **[CONCLUÍDO]**
- [x] **Task 4.2:** Escrever as arquiteturas PyTorch: usar um modelo de *embeddings* de texto, o `GraphSAGE` e a `Bi-GRU`. Inicializar a rede com os pesos base do dataset **Twibot-20** (Transfer Learning) em vez de pesos aleatórios. **[CONCLUÍDO]**
- [x] **Task 4.3:** Integrar o Stream Processor com a API: recolher features reais do Redis e do Neo4j em tempo real, submeter à inferência, e aplicar a mitigação baseada no Score. **[CONCLUÍDO]**

---

### Marco 5: MLOps e Pipeline Científico (O Diferencial de Produção)
*Objetivo: Treino robusto, revisão humana inteligente e imunidade a Zero-Day Bots.*

- [x] **Task 5.1 (Heurísticas de Cresci):** Desenvolver o `labeler_heuristico.py`. Aplicar regras baseadas em literatura (Entropia Temporal, Rácio Following/Followers, Densidade de URLs) para gerar a `"observed_label"` e construir o *Dataset V1*. **[CONCLUÍDO]**
- [x] **Task 5.2 (Active Learning):** Criar uma CLI de revisão manual (`manual_review.py`) que extraia apenas 50 eventos onde o modelo esteve altamente incerto ($P(Bot) \approx 0.50$). Injetar estas classificações humanas no re-treino. **[CONCLUÍDO]**
- [x] **Task 5.3 (CALEB - Data Augmentation):** Implementar o gerador de dados sintéticos adversariais (CGAN) para simular novas táticas de evasão (inserir pausas, diluir links) e expandir o dataset de treino. **[CONCLUÍDO]**
- [x] **Task 5.4 (Hot-Swap):** Escrever o script de re-treino `train.py` e implementar no FastAPI o carregamento do novo modelo `.pt` sem *downtime*. **[CONCLUÍDO]**

---

### Marco 6: Resiliência em Larga Escala & Explicabilidade (Pós-Marco 5)
*Objetivo: Garantir tolerância a falhas em produção, explicabilidade das decisões de banimento e Shadow Deployments.*

- [x] **Task 6.1 (Fila de Mensagens Mortas - DLQ)**: Implementar uma Dead-Letter Queue (DLQ) no processador. Se uma mensagem falhar no parsing ou estourar o timeout da API, enviá-la para o tópico `user_actions_dlq` em vez de travar a partição do Kafka. **[CONCLUÍDO]**
- [x] **Task 6.2 (Explainability Head - SHAP/Feature Importances)**: Implementar na resposta do FastAPI os fatores de contribuição da decisão (ex: "Razão seguidores/amigos contribuiu 45% para a classificação de Bot"). Essencial para suporte a clientes banidos erroneamente. **[CONCLUÍDO]**
- [x] **Task 6.3 (Shadow Deployment & Shadow Mode)**: Adicionar suporte ao FastAPI para executar o modelo retreinado em modo "sombra" (Shadow Mode), comparando suas predições ao vivo com o baseline de produção e gerando relatórios de divergência de métricas antes do Hot-Swap definitivo. **[CONCLUÍDO]**