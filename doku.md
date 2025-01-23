# Code-Dokumentation

## Einführung

Dieses Dokument beschreibt den Code zur Verarbeitung und Simulation von Daten aus einer CSV-Datei, die in einem neuronalen Netzwerk verwendet werden. Der Code umfasst Funktionen zur Initialisierung des Netzwerks, zur Verarbeitung der CSV-Datei, zur Simulation des Lernprozesses und zur Speicherung und Laden des Modells.

## Abhängigkeiten

- `pandas`: Zur Verarbeitung von CSV-Dateien.
- `numpy`: Für numerische Operationen.
- `random`: Für zufällige Operationen.
- `tqdm`: Für Fortschrittsanzeigen.
- `tkinter`: Für die grafische Benutzeroberfläche.
- `seaborn`: Für die Visualisierung.
- `networkx`: Für die Erstellung und Analyse von Graphen.
- `json`: Für die Speicherung und das Laden von Modellen.
- `os`: Für Dateioperationen.
- `time`: Für Zeitmessungen.
- `torch`: Für neuronale Netzwerke.
- `threading`: Für die Verwaltung von Threads.
- `logging`: Für die Protokollierung.
- `sqlite3`: Für die Verwendung von SQLite-Datenbanken.
- `dask.dataframe`: Für die parallele Verarbeitung von Daten.

## Konfiguration des Loggers

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Globale Variablen

- `initialized`: Überprüft, ob das Netzwerk initialisiert wurde.
- `category_nodes`: Liste der Knoten im Netzwerk.
- `questions`: Liste der Fragen.
- `model_saved`: Schutzvariable, um zu überprüfen, ob das Modell gespeichert wurde.

## Überprüfen, ob der Ordner existiert

```python
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

## Funktionen

### Funktion zum Aufteilen der CSV-Datei

```python
def split_csv(filename, chunk_size=1000, output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_iter = pd.read_csv(filename, chunksize=chunk_size)
    for i, chunk in enumerate(chunk_iter):
        chunk.to_csv(os.path.join(output_dir, f"data_part_{i}.csv"), index=False)
        logging.info(f"Chunk {i} mit {len(chunk)} Zeilen gespeichert.")
```

### Verbesserung 1: Verstärkung der Verbindungen bei häufig gestellten Fragen

```python
def strengthen_question_connection(category_nodes, question, category):
    category_node = next((node for node in category_nodes if node.label == category), None)
    if category_node:
        for conn in category_node.connections:
            if conn.target_node.label == question:
                old_weight = conn.weight
                conn.weight += 0.1  # Verstärkung der Verbindung
                conn.weight = np.clip(conn.weight, 0, 1.0)
                logging.info(f"Verstärkte Verbindung für Frage '{question}' in Kategorie '{category}': {old_weight:.4f} -> {conn.weight:.4f}")
```

### Verbesserung 2: Erweiterte Hebb'sche Lernregel zur besseren Zuordnung von Fragen

```python
def enhanced_hebbian_learning(node, target_node, learning_rate=0.2, decay_factor=0.01):
    old_weight = None
    for conn in node.connections:
        if conn.target_node == target_node:
            old_weight = conn.weight
            conn.weight += learning_rate * node.activation * target_node.activation
            conn.weight = np.clip(conn.weight - decay_factor * conn.weight, 0, 1.0)
            break

    if old_weight is not None:
        logging.info(f"Hebb'sches Lernen angewendet: Gewicht {old_weight:.4f} -> {conn.weight:.4f}")
```

### Verbesserung 3: Simulation der Frageverarbeitung im Netzwerk

```python
def simulate_question_answering(category_nodes, question, questions):
    category = next((q['category'] for q in questions if q['question'] == question), None)
    if not category:
        logging.warning(f"Frage '{question}' nicht gefunden!")
        return None

    category_node = next((node for node in category_nodes if node.label == category), None)
    if category_node:
        propagate_signal(category_node, input_signal=0.9, emotion_weights={}, emotional_state=1.0)
        activation = category_node.activation
        if activation is None or activation <= 0:
            logging.warning(f"Kategorie '{category}' hat eine ungültige Aktivierung: {activation}")
            return 0.0  # Rückgabe von 0, falls die Aktivierung fehlschlägt
        logging.info(f"Verarbeite Frage: '{question}' → Kategorie: '{category}' mit Aktivierung {activation:.4f}")
        return activation  # Entfernte doppelte Logging-Ausgabe
    else:
        logging.warning(f"Kategorie '{category}' nicht im Netzwerk gefunden. Die Kategorie wird neu hinzugefügt!")
        return 0.0
```

### Verbesserung 4: Finden der besten passenden Frage zur Benutzeranfrage

```python
def find_question_by_keyword(questions, keyword):
    matching_questions = [q for q in questions if keyword.lower() in q['question'].lower()]
    return matching_questions if matching_questions else None
```

### Verbesserung 5: Suche nach der ähnlichsten Frage basierend auf einfachen Ähnlichkeitsmetriken

```python
def find_similar_question(questions, query):
    from difflib import get_close_matches
    question_texts = [q['question'] for q in questions]
    closest_matches = get_close_matches(query, question_texts, n=1, cutoff=0.6)

    if closest_matches:
        matched_question = next((q for q in questions if q['question'] == closest_matches[0]), None)
        return matched_question
    else:
        return {"question": "Keine passende Frage gefunden", "category": "Unbekannt"}
```

### Verbesserung 6: Testfunktion zur Überprüfung des Modells

```python
def test_model(category_nodes, questions, query):
    matched_question = find_question_by_keyword(questions, query)
    if matched_question:
        logging.info(f"Gefundene Frage: {matched_question[0]['question']} -> Kategorie: {matched_question[0]['category']}")
        simulate_question_answering(category_nodes, matched_question[0]['question'], questions)
    else:
        logging.warning("Keine passende Frage gefunden.")

    similarity_question = find_similar_question(questions, query)
    logging.info(f"Ähnlichste Frage: {similarity_question['question']} -> Kategorie: {similarity_question['category']}")
```

### NetworkX-Funktionen für kausale Graphen

```python
def build_causal_graph(category_nodes):
    G = nx.DiGraph()
    for node in category_nodes:
        G.add_node(node.label)
        for conn in node.connections:
            G.add_edge(node.label, conn.target_node.label, weight=conn.weight)
    return G

def analyze_causality_multiple(G, num_pairs=3):
    if len(G.nodes) < 2:
        logging.warning("Graph enthält nicht genügend Knoten für eine Analyse.")
        return

    for _ in range(num_pairs):
        start_node, target_node = random.sample(G.nodes, 2)
        logging.info(f"Analysiere kausale Pfade von '{start_node}' nach '{target_node}'")

        try:
            paths = list(nx.all_simple_paths(G, source=start_node, target=target_node))
            if paths:
                for path in paths:
                    logging.info(f"Kausaler Pfad: {' -> '.join(path)}")
            else:
                logging.info(f"Kein Pfad gefunden von '{start_node}' nach '{target_node}'")
        except nx.NetworkXNoPath:
            logging.warning(f"Kein direkter Pfad zwischen '{start_node}' und '{target_node}' gefunden.")

def analyze_node_influence(G):
    influence_scores = nx.pagerank(G, alpha=0.85)
    sorted_influences = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_influences:
        logging.info(f"Knoten: {node}, Einfluss: {score:.4f}")
```

### Funktion für Interventionen basierend auf Pearl's Do-Operator

```python
def do_intervention(node, new_value):
    logging.info(f"Intervention: Setze {node.label} auf {new_value}")
    node.activation = new_value
    for conn in node.connections:
        conn.target_node.activation += node.activation * conn.weight
```

### Kontextabhängiges Lernen verstärken

```python
def contextual_causal_analysis(node, context_factors, learning_rate=0.1):
    context_factor = context_factors.get(node.label, 1.0)
    if node.activation > 0.8 and context_factor > 1.0:
        logging.info(f"Kausale Beziehung verstärkt für {node.label} aufgrund des Kontextes.")
        for conn in node.connections:
            conn.weight += learning_rate * context_factor
            conn.weight = np.clip(conn.weight, 0, 1.0)
            logging.info(f"Gewicht aktualisiert: {node.label} → {conn.target_node.label}, Gewicht: {conn.weight:.4f}")
```

### PyTorch-Modell für kausale Inferenz

```python
class CausalInferenceNN(nn.Module):
    def __init__(self):
        super(CausalInferenceNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### Debugging-Funktion

```python
def debug_connections(category_nodes):
    start_time = time.time()
    for node in category_nodes:
        logging.info(f"Knoten: {node.label}")
        for conn in node.connections:
            logging.info(f" Verbindung zu: {conn.target_node.label}, Gewicht: {conn.weight}")
    end_time = time.time()
    logging.info(f"debug_connections Ausführungszeit: {end_time - start_time:.4f} Sekunden")
```

### Hilfsfunktionen

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def add_activation_noise(activation, noise_level=0.1):
    noise = np.random.normal(0, noise_level)
    return np.clip(activation + noise, 0.0, 1.0)

def decay_weights(category_nodes, decay_rate=0.002, forgetting_curve=0.95):
    for node in category_nodes:
        for conn in node.connections:
            conn.weight *= (1 - decay_rate) * forgetting_curve

def reward_connections(category_nodes, target_category, reward_factor=0.1):
    for node in category_nodes:
        if node.label == target_category:
            for conn in node.connections:
                conn.weight += reward_factor
                conn.weight = np.clip(conn.weight, 0, 1.0)

def apply_emotion_weight(activation, category_label, emotion_weights, emotional_state=1.0):
    emotion_factor = emotion_weights.get(category_label, 1.0) * emotional_state
    return activation * emotion_factor

def generate_simulated_answers(data, personality_distributions):
    simulated_answers = []
    for _, row in data.iterrows():
        category = row['Kategorie']
        mean = personality_distributions.get(category, 0.5)
        simulated_answer = np.clip(np.random.normal(mean, 0.2), 0.0, 1.0)
        simulated_answers.append(simulated_answer)
    return simulated_answers

def social_influence(category_nodes, social_network, influence_factor=0.1):
    for node in category_nodes:
        for conn in node.connections:
            social_impact = sum([social_network.get(conn.target_node.label, 0)]) * influence_factor
            conn.weight += social_impact
            conn.weight = np.clip(conn.weight, 0, 1.0)

def update_emotional_state(emotional_state, emotional_change_rate=0.02):
    emotional_state += np.random.normal(0, emotional_change_rate)
    return np.clip(emotional_state, 0.7, 1.5)

def apply_contextual_factors(activation, node, context_factors):
    context_factor = context_factors.get(node.label, 1.0)
    return activation * context_factor * random.uniform(0.9, 1.1)

def long_term_memory(category_nodes, long_term_factor=0.01):
    for node in category_nodes:
        for conn in node.connections:
            conn.weight += long_term_factor * conn.weight
            conn.weight = np.clip(conn.weight, 0, 1.0)

def hebbian_learning(node, learning_rate=0.3, weight_limit=1.0, reg_factor=0.005):
    for connection in node.connections:
        old_weight = connection.weight
        connection.weight += learning_rate * node.activation * connection.target_node.activation
        connection.weight = np.clip(connection.weight, -weight_limit, weight_limit)
        connection.weight -= reg_factor * connection.weight
        node.activation_history.append(node.activation)  # Aktivierung speichern
        connection.target_node.activation_history.append(connection.target_node.activation)
        logging.info(f"Hebb'sches Lernen: Gewicht von {old_weight:.4f} auf {connection.weight:.4f} erhöht")
```

### Klassen für Netzwerkstruktur

```python
class Connection:
    def __init__(self, target_node, weight=None):
        self.target_node = target_node
        self.weight = weight if weight is not None else random.uniform(0.1, 1.0)

class Node:
    def __init__(self, label):
        self.label = label
        self.connections = []
        self.activation = 0.0
        self.activation_history = []

    def add_connection(self, target_node, weight=None):
        self.connections.append(Connection(target_node, weight))

    def save_state(self):
        return {
            "label": self.label,
            "activation": self.activation,
            "activation_history": self.activation_history,
            "connections": [{"target": conn.target_node.label, "weight": conn.weight} for conn in self.connections]
        }

    @staticmethod
    def load_state(state, nodes_dict):
        node = Node(state["label"])
        node.activation = state["activation"]
        node.activation_history = state["activation_history"]
        for conn_state in state["connections"]:
            target_node = nodes_dict[conn_state["target"]]
            connection = Connection(target_node, conn_state["weight"])
            node.connections.append(connection)
        return node

class MemoryNode(Node):
    def __init__(self, label, memory_type="short_term"):
        super().__init__(label)
        self.memory_type = memory_type
        self.retention_time = {"short_term": 5, "mid_term": 20, "long_term": 100}[memory_type]
        self.time_in_memory = 0

    def decay(self, decay_rate, context_factors, emotional_state):
        context_factor = context_factors.get(self.label, 1.0)
        emotional_factor = emotional_state
        for conn in self.connections:
            if self.memory_type == "short_term":
                conn.weight *= (1 - decay_rate * 2 * context_factor * emotional_factor)
            elif self.memory_type == "mid_term":
                conn.weight *= (1 - decay_rate * context_factor * emotional_factor)
            elif self.memory_type == "long_term":
                conn.weight *= (1 - decay_rate * 0.5 * context_factor * emotional_factor)

    def promote(self, activation_threshold=0.7):
        if len(self.activation_history) == 0:
            return
        if self.memory_type == "short_term" and np.mean(self.activation_history[-5:]) > activation_threshold:
            self.memory_type = "mid_term"
            self.retention_time = 20
        elif self.memory_type == "mid_term" and np.mean(self.activation_history[-20:]) > activation_threshold:
            self.memory_type = "long_term"
            self.retention_time = 100

class CortexCreativus(Node):
    def __init__(self, label):
        super().__init__(label)

    def generate_new_ideas(self, category_nodes):
        new_ideas = []
        for node in category_nodes:
            if node.activation > 0.5:
                new_idea = f"New idea based on {node.label} with activation {node.activation}"
                new_ideas.append(new_idea)
        return new_ideas

class SimulatrixNeuralis(Node):
    def __init__(self, label):
        super().__init__(label)

    def simulate_scenarios(self, category_nodes):
        scenarios = []
        for node in category_nodes:
            if node.activation > 0.5:
                scenario = f"Simulated scenario based on {node.label} with activation {node.activation}"
                scenarios.append(scenario)
        return scenarios

class CortexCriticus(Node):
    def __init__(self, label):
        super().__init__(label)

    def evaluate_ideas(self, ideas):
        evaluated_ideas = []
        for idea in ideas:
            evaluation_score = random.uniform(0, 1)
            evaluation = f"Evaluated idea: {idea} - Score: {evaluation_score}"
            evaluated_ideas.append(evaluation)
        return evaluated_ideas

class LimbusAffectus(Node):
    def __init__(self, label):
        super().__init__(label)

    def apply_emotion_weight(self, ideas, emotional_state):
        weighted_ideas = []
        for idea in ideas:
            weighted_idea = f"Emotionally weighted idea: {idea} - Weight: {emotional_state}"
            weighted_ideas.append(weighted_idea)
        return weighted_ideas

class MetaCognitio(Node):
    def __init__(self, label):
        super().__init__(label)

    def optimize_system(self, category_nodes):
        for node in category_nodes:
            node.activation *= random.uniform(0.9, 1.1)

class CortexSocialis(Node):
    def __init__(self, label):
        super().__init__(label)

    def simulate_social_interactions(self, category_nodes):
        interactions = []
        for node in category_nodes:
            if node.activation > 0.5:
                interaction = f"Simulated social interaction based on {node.label} with activation {node.activation}"
                interactions.append(interaction)
        return interactions

def connect_new_brains_to_network(category_nodes, new_brains):
    for brain in new_brains:
        for node in category_nodes:
            brain.add_connection(node)
            node.add_connection(brain)
```

### Netzwerk-Initialisierung

```python
def initialize_quiz_network(categories):
    try:
        category_nodes = [Node(c) for c in categories]
        for node in category_nodes:
            for target_node in category_nodes:
                if node != target_node:
                    node.add_connection(target_node)
                    logging.debug(f"Verbindung hinzugefügt: {node.label} → {target_node.label}")
        debug_connections(category_nodes)
        for node in category_nodes:
            logging.info(f"Knoten erstellt: {node.label}")
            for conn in node.connections:
                logging.info(f"  → Verbindung zu {conn.target_node.label} mit Gewicht {conn.weight:.4f}")
        return category_nodes
    except Exception as e:
        logging.error(f"Fehler bei der Netzwerk-Initialisierung: {e}")
        return []
```

### Signalpropagation

```python
def propagate_signal(node, input_signal, emotion_weights, emotional_state=1.0, context_factors=None):
    node.activation = add_activation_noise(sigmoid(input_signal * random.uniform(0.8, 1.2)))
    node.activation_history.append(node.activation)  # Aktivierung speichern
    node.activation = apply_emotion_weight(node.activation, node.label, emotion_weights, emotional_state)
    if context_factors:
        node.activation = apply_contextual_factors(node.activation, node, context_factors)
    logging.info(f"Signalpropagation für {node.label}: Eingangssignal {input_signal:.4f}")
    for connection in node.connections:
        logging.info(f"  → Signal an {connection.target_node.label} mit Gewicht {connection.weight:.4f}")
        connection.target_node.activation += node.activation * connection.weight

def propagate_signal_with_memory(node, input_signal, category_nodes, memory_nodes, context_factors, emotional_state):
    node.activation = add_activation_noise(sigmoid(input_signal))
    node.activation_history.append(node.activation)
    for connection in node.connections:
        connection.target_node.activation += node.activation * connection.weight
    for memory_node in memory_nodes:
        memory_node.time_in_memory += 1
        memory_node.promote()
```

### Simulation mit Anpassungen

```python
def simulate_learning(data, category_nodes, personality_distributions, epochs=1, learning_rate=0.8, reward_interval=5, decay_rate=0.002, emotional_state=1.0, context_factors=None):
    if context_factors is None:
        context_factors = {}

    weights_history = {f"{node.label} → {conn.target_node.label}": [] for node in category_nodes for conn in node.connections}
    activation_history = {node.label: [] for node in category_nodes}
    question_nodes = []

    for idx, row in data.iterrows():
        q_node = Node(row['Frage'])
        question_nodes.append(q_node)
        category_label = row['Kategorie'].strip()
        category_node = next((c for c in category_nodes if c.label == category_label), None)
        if category_node:
            q_node.add_connection(category_node)
            logging.debug(f"Verbindung hinzugefügt: {q_node.label} → {category_node.label}")
        else:
            logging.warning(f"Warnung: Kategorie '{category_label}' nicht gefunden für Frage '{row['Frage']}'.")

    emotion_weights = {category: 1.0 for category in data['Kategorie'].unique()}
    social_network = {category: random.uniform(0.1, 1.0) for category in data['Kategorie'].unique()}

    for epoch in range(epochs):
        logging.info(f"\n--- Epoche {epoch + 1} ---")
        simulated_answers = generate_simulated_answers(data, personality_distributions)

        for node in category_nodes:
            node.activation_sum = 0.0
            node.activation_count = 0

        for node in category_nodes:
            propagate_signal(node, random.uniform(0.1, 0.9), emotion_weights, emotional_state, context_factors)
            node.activation_history.append(node.activation)  # Aktivierung speichern

        for idx, q_node in enumerate(question_nodes):
            for node in category_nodes + question_nodes:
                node.activation = 0.0
            answer = simulated_answers[idx]
            propagate_signal(q_node, answer, emotion_weights, emotional_state, context_factors)
            q_node.activation_history.append(q_node.activation)  # Aktivierung speichern
            hebbian_learning(q_node, learning_rate)

            for node in category_nodes:
                node.activation_sum += node.activation
                if node.activation > 0:
                    node.activation_count += 1

            for node in category_nodes:
                for conn in node.connections:
                    weights_history[f"{node.label} → {conn.target_node.label}"].append(conn.weight)
                    logging.debug(f"Gewicht aktualisiert: {node.label} → {conn.target_node.label}, Gewicht: {conn.weight}")

            # Kausalitätsverstärkung anwenden
            contextual_causal_analysis(q_node, context_factors, learning_rate)

        for node in category_nodes:
            if node.activation_count > 0:
                mean_activation = node.activation_sum / node.activation_count
                activation_history[node.label].append(mean_activation)
                logging.info(f"Durchschnittliche Aktivierung für Knoten {node.label}: {mean_activation:.4f}")
            else:
                activation_history[node.label].append(0.0)
                logging.info(f"Knoten {node.label} wurde in dieser Epoche nicht aktiviert.")

        if (epoch + 1) % reward_interval == 0:
            target_category = random.choice(data['Kategorie'].unique())
            reward_connections(category_nodes, target_category=target_category)

        decay_weights(category_nodes, decay_rate=decay_rate)
        social_influence(category_nodes, social_network)

    logging.info("Simulation abgeschlossen. Ergebnisse werden analysiert...")
    return activation_history, weights_history
```

### Simulation mit mehrstufigem Gedächtnis

```python
def simulate_multilevel_memory(data, category_nodes, personality_distributions, epochs=1):
    short_term_memory = [MemoryNode(c, "short_term") for c in category_nodes]
    mid_term_memory = []
    long_term_memory = []
    memory_nodes = short_term_memory + mid_term_memory + long_term_memory
    context_factors = {question: random.uniform(0.9, 1.1) for question in data['Frage'].unique()}
    emotional_state = 1.0
    for epoch in range(epochs):
        logging.info(f"\n--- Epoche {epoch + 1} ---")
        for node in short_term_memory:
            input_signal = random.uniform(0.1, 1.0)
            propagate_signal_with_memory(node, input_signal, category_nodes, memory_nodes, context_factors, emotional_state)
        for memory_node in memory_nodes:
            memory_node.decay(decay_rate=0.01, context_factors=context_factors, emotional_state=emotional_state)
        for memory_node in memory_nodes:
            memory_node.promote()
        short_term_memory, mid_term_memory, long_term_memory = update_memory_stages(memory_nodes)
        logging.info(f"Epoche {epoch + 1}: Kurzzeit {len(short_term_memory)}, Mittelzeit {len(mid_term_memory)}, Langzeit {len(long_term_memory)}")
    return short_term_memory, mid_term_memory, long_term_memory

def update_memory_stages(memory_nodes):
    short_term_memory = [node for node in memory_nodes if node.memory_type == "short_term"]
    mid_term_memory = [node for node in memory_nodes if node.memory_type == "mid_term"]
    long_term_memory = [node for node in memory_nodes if node.memory_type == "long_term"]
    return short_term_memory, mid_term_memory, long_term_memory
```

### Plot-Funktionen

```python
def plot_activation_history(activation_history, filename="activation_history.png"):
    if not activation_history:
        logging.warning("No activation history to plot")
        return
    plt.figure(figsize=(12, 8))
    for label, activations in activation_history.items():
        if len(activations) > 0:
            plt.plot(range(1, len(activations) + 1), activations, label=label)
    plt.title("Entwicklung der Aktivierungen während des Lernens")
    plt.xlabel("Epoche")
    plt.ylabel("Aktivierung")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Plot gespeichert unter: {os.path.join(output_dir, filename)}")

def plot_dynamics(activation_history, weights_history, filename="dynamics.png"):
    if not weights_history:
        logging.error("weights_history ist leer.")
        return

    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    for label, activations in activation_history.items():
        if len(activations) > 0:
            plt.plot(range(1, len(activations) + 1), activations, label=label)
    plt.title("Entwicklung der Aktivierungen während des Lernens")
    plt.xlabel("Epoche")
    plt.ylabel("Aktivierung")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for label, weights in weights_history.items():
        if len(weights) > 0:
            plt.plot(range(1, len(weights) + 1), weights, label=label, alpha=0.7)
    plt.title("Entwicklung der Verbindungsgewichte während des Lernens")
    plt.xlabel("Epoche")
    plt.ylabel("Gewicht")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Plot gespeichert unter: {os.path.join(output_dir, filename)}")

def plot_memory_distribution(short_term_memory, mid_term_memory, long_term_memory, filename="memory_distribution.png"):
    counts = [len(short_term_memory), len(mid_term_memory), len(long_term_memory)]
    labels = ["Kurzfristig", "Mittelfristig", "Langfristig"]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=["red", "blue", "green"])
    plt.title("Verteilung der Gedächtnisknoten")
    plt.ylabel("Anzahl der Knoten")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Plot gespeichert unter: {os.path.join(output_dir, filename)}")

def plot_activation_heatmap(activation_history, filename="activation_heatmap.png"):
    if not activation_history:
        logging.warning("No activation history to plot")
        return

    min_length = min(len(activations) for activations in activation_history.values())
    truncated_activations = {key: values[:min_length] for key, values in activation_history.items()}

    plt.figure(figsize=(12, 8))
    heatmap_data = np.array([activations for activations in truncated_activations.values()])

    if heatmap_data.size == 0:
        logging.error("Heatmap-Daten sind leer. Überprüfen Sie die Aktivierungshistorie.")
        return

    sns.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=truncated_activations.keys(), yticklabels=False)
    plt.title("Heatmap der Aktivierungswerte")
    plt.xlabel("Kategorie")
    plt.ylabel("Epoche")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Plot gespeichert unter: {os.path.join(output_dir, filename)}")

def plot_network_topology(category_nodes, new_brains, filename="network_topology.png"):
    G = nx.DiGraph()
    for node in category_nodes:
        G.add_node(node.label)
        for conn in node.connections:
            G.add_edge(node.label, conn.target_node.label, weight=conn.weight)
    for brain in new_brains:
        G.add_node(brain.label, color='red')
        for conn in brain.connections:
            G.add_edge(brain.label, conn.target_node.label, weight=conn.weight)

    pos = nx.spring_layout(G)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Netzwerktopologie")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Plot gespeichert unter: {os.path.join(output_dir, filename)}")
```

### Modell speichern und laden

```python
def save_model(category_nodes, filename="model.json"):
    model_data = {
        "nodes": [node.save_state() for node in category_nodes]
    }
    with open(filename, "w") as file:
        json.dump(model_data, file, indent=4)
    logging.info(f"Modell gespeichert in {filename}")

def save_model_with_questions_and_answers(category_nodes, questions, filename="model_with_qa.json"):
    global model_saved
    logging.info("Starte Speichern des Modells...")

    # Überprüfen, ob Änderungen vorgenommen wurden
    current_model_data = {
        "nodes": [node.save_state() for node in category_nodes],
        "questions": questions
    }

    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                existing_model_data = json.load(file)
                if existing_model_data == current_model_data:
                    logging.info("Keine Änderungen erkannt, erneutes Speichern übersprungen.")
                    return
        except Exception as e:
            logging.warning(f"Fehler beim Überprüfen des vorhandenen Modells: {e}")

    # Speichern des aktualisierten Modells
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(current_model_data, file, indent=4)
        logging.info(f"Modell erfolgreich gespeichert unter {filename}.")
        model_saved = True  # Setze auf True nach erfolgreichem Speichern
    except Exception as e:
        logging.error(f"Fehler beim Speichern des Modells: {e}")

def load_model_with_questions_and_answers(filename="model_with_qa.json"):
    global initialized
    if initialized:
        logging.info("Modell bereits initialisiert.")
        return None, None

    if not os.path.exists(filename):
        logging.warning(f"Datei {filename} nicht gefunden. Netzwerk wird initialisiert.")
        return None, None

    try:
        with open(filename, "r", encoding="utf-8") as file:
            model_data = json.load(file)

        nodes_dict = {node_data["label"]: Node(node_data["label"]) for node_data in model_data["nodes"]}

        for node_data in model_data["nodes"]:
            node = nodes_dict[node_data["label"]]
            node.activation = node_data.get("activation", 0.0)
            for conn_state in node_data["connections"]:
                target_node = nodes_dict.get(conn_state["target"])
                if target_node:
                    node.add_connection(target_node, conn_state["weight"])

        questions = model_data.get("questions", [])
        logging.info(f"Modell geladen mit {len(nodes_dict)} Knoten und {len(questions)} Fragen")
        initialized = True
        return list(nodes_dict.values()), questions

    except json.JSONDecodeError as e:
        logging.error(f"Fehler beim Parsen der JSON-Datei: {e}")
        return None, None
```

### Fragen aktualisieren

```python
def update_questions_with_answers(filename="model_with_qa.json"):
    with open(filename, "r") as file:
        model_data = json.load(file)

    for question in model_data["questions"]:
        if "answer" not in question:
            question["answer"] = input(f"Gib die Antwort für: '{question['question']}': ")

    with open(filename, "w") as file:
        json.dump(model_data, file, indent=4)
    logging.info(f"Fragen wurden mit Antworten aktualisiert und gespeichert in {filename}")
```

### Beste Antwort finden

```python
def find_best_answer(category_nodes, questions, query):
    matched_question = find_similar_question(questions, query)
    if matched_question:
        logging.info(f"Gefundene Frage: {matched_question['question']} -> Kategorie: {matched_question['category']}")
        answer = matched_question.get("answer", "Keine Antwort verfügbar")
        logging.info(f"Antwort: {answer}")
        return answer
    else:
        logging.warning("Keine passende Frage gefunden.")
        return None
```

### Dashboard erstellen

```python
def create_dashboard(category_nodes, activation_history, short_term_memory, mid_term_memory, long_term_memory):
    root = tk.Tk()
    root.title("Psyco Dashboard")

    # Anzeige der Aktivierungshistorie
    activation_frame = ttk.Frame(root, padding="10")
    activation_frame.pack(fill=tk.BOTH, expand=True)
    activation_label = ttk.Label(activation_frame, text="Aktivierungshistorie")
    activation_label.pack()
    if activation_history:
        for label, activations in activation_history.items():
            fig, ax = plt.subplots()
            ax.plot(range(1, len(activations) + 1), activations)
            ax.set_title(label)
            canvas = FigureCanvasTkAgg(fig, master=activation_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
    else:
        no_data_label = ttk.Label(activation_frame, text="Keine Aktivierungshistorie verfügbar.")
        no_data_label.pack()

    # Anzeige der Gedächtnisverteilung
    memory_frame = ttk.Frame(root, padding="10")
    memory_frame.pack(fill=tk.BOTH, expand=True)
    memory_label = ttk.Label(memory_frame, text="Gedächtnisverteilung")
    memory_label.pack()
    memory_counts = [len(short_term_memory), len(mid_term_memory), len(long_term_memory)]
    labels = ["Kurzfristig", "Mittelfristig", "Langfristig"]
    fig, ax = plt.subplots()
    ax.bar(labels, memory_counts, color=["red", "blue", "green"])
    ax.set_title("Verteilung der Gedächtnisknoten")
    ax.set_ylabel("Anzahl der Knoten")
    canvas = FigureCanvasTkAgg(fig, master=memory_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Anzeige der Netzwerktopologie
    topology_frame = ttk.Frame(root, padding="10")
    topology_frame.pack(fill=tk.BOTH, expand=True)
    topology_label = ttk.Label(topology_frame, text="Netzwerktopologie")
    topology_label.pack()
    G = nx.DiGraph()
    for node in category_nodes:
        G.add_node(node.label)
        for conn in node.connections:
            G.add_edge(node.label, conn.target_node.label, weight=conn.weight)
    pos = nx.spring_layout(G)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    node_colors = ['skyblue' for _ in G.nodes()]
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, font_weight="bold", edge_color="gray", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("Netzwerktopologie")
    canvas = FigureCanvasTkAgg(fig, master=topology_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Anzeige der Heatmap der Aktivierungswerte
    heatmap_frame = ttk.Frame(root, padding="10")
    heatmap_frame.pack(fill=tk.BOTH, expand=True)
    heatmap_label = ttk.Label(heatmap_frame, text="Heatmap der Aktivierungswerte")
    heatmap_label.pack()
    if activation_history:
        min_length = min(len(activations) for activations in activation_history.values())
        truncated_activations = {key: values[:min_length] for key, values in activation_history.items()}
        heatmap_data = np.array([activations for activations in truncated_activations.values()])
        if heatmap_data.size > 0:
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=truncated_activations.keys(), yticklabels=False, ax=ax)
            ax.set_title("Heatmap der Aktivierungswerte")
            ax.set_xlabel("Kategorie")
            ax.set_ylabel("Epoche")
            canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            no_data_label = ttk.Label(heatmap_frame, text="Heatmap-Daten sind leer. Überprüfen Sie die Aktivierungshistorie.")
            no_data_label.pack()
    else:
        no_data_label = ttk.Label(heatmap_frame, text="Keine Aktivierungshistorie verfügbar.")
        no_data_label.pack()

    root.mainloop()
```

### CSV-Datei verarbeiten

```python
def process_csv_in_chunks(filename, chunk_size=10000):
    global category_nodes, questions
    logging.info(f"Beginne Verarbeitung der Datei: {filename}")

    try:
        # Test, ob die Datei existiert
        if not os.path.exists(filename):
            logging.error(f"Datei {filename} nicht gefunden.")
            return None

        all_chunks = []
        for chunk in pd.read_csv(filename, chunksize=chunk_size, encoding="utf-8", on_bad_lines='skip'):
            logging.info(f"Chunk mit {len(chunk)} Zeilen gelesen.")
            if 'Frage' not in chunk.columns or 'Kategorie' not in chunk.columns or 'Antwort' not in chunk.columns:
                logging.error("CSV-Datei enthält nicht die erwarteten Spalten: 'Frage', 'Kategorie', 'Antwort'")
                return None

            all_chunks.append(chunk)

        data = pd.concat(all_chunks, ignore_index=True)
        logging.info(f"Alle Chunks erfolgreich verarbeitet. Gesamtzeilen: {len(data)}")

        return data

    except pd.errors.EmptyDataError:
        logging.error("CSV-Datei ist leer.")
    except pd.errors.ParserError as e:
        logging.error(f"Parsing-Fehler in CSV-Datei: {e}")
    except Exception as e:
        logging.error(f"Unerwarteter Fehler beim Verarbeiten der Datei: {e}")

    return None
```

### Einzelne Einträge verarbeiten

```python
def process_single_entry(question, category, answer):
    global category_nodes, questions

    # Sicherstellen, dass die globalen Variablen initialisiert sind
    if category_nodes is None:
        category_nodes = []
        logging.warning("Kategorie-Knotenliste war None, wurde nun initialisiert.")

    if questions is None:
        questions = []
        logging.warning("Fragenliste war None, wurde nun initialisiert.")

    # Überprüfen, ob die Kategorie bereits vorhanden ist
    if not any(node.label == category for node in category_nodes):
        category_nodes.append(Node(category))
        logging.info(f"Neue Kategorie '{category}' dem Netzwerk hinzugefügt.")

    # Frage, Kategorie und Antwort zur Liste hinzufügen
    questions.append({"question": question, "category": category, "answer": answer})
    logging.info(f"Neue Frage hinzugefügt: '{question}' -> Kategorie: '{category}'")
```

### CSV-Datei mit Dask verarbeiten

```python
def process_csv_with_dask(filename, chunk_size=10000):
    try:
        ddf = dd.read_csv(filename, blocksize=chunk_size)
        ddf = ddf.astype({'Kategorie': 'category'})

        for row in ddf.itertuples(index=False, name=None):
            process_single_entry(row[0], row[1], row[2])

        logging.info("Alle Chunks erfolgreich mit Dask verarbeitet.")
    except Exception as e:
        logging.error(f"Fehler beim Verarbeiten der Datei mit Dask: {e}")
```

### In SQLite speichern

```python
def save_to_sqlite(filename, db_name="dataset.db"):
    conn = sqlite3.connect(db_name)
    chunk_iter = pd.read_csv(filename, chunksize=10000)
    for chunk in chunk_iter:
        chunk.to_sql("qa_data", conn, if_exists="append", index=False)
        logging.info(f"Chunk mit {len(chunk)} Zeilen gespeichert.")
    conn.close()
    logging.info("CSV-Daten wurden erfolgreich in SQLite gespeichert.")
```

### Aus SQLite laden

```python
def load_from_sqlite(db_name="dataset.db"):
    conn = sqlite3.connect(db_name)
    query = "SELECT Frage, Kategorie, Antwort FROM qa_data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data
```

### Teilmodell speichern

```python
def save_partial_model(filename="partial_model.json"):
    model_data = {
        "nodes": [node.save_state() for node in category_nodes],
        "questions": questions
    }
    with open(filename, "w") as file:
        json.dump(model_data, file, indent=4)
    logging.info("Teilmodell gespeichert.")
```

### CSV-Datei faul laden

```python
def lazy_load_csv(filename, chunk_size=10000):
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield row['Frage'], row['Kategorie'], row['Antwort']
```

### Hauptfunktion

```python
def main():
    start_time = time.time()
    category_nodes, questions = load_model_with_questions_and_answers("model_with_qa.json")

    if category_nodes is None:
        csv_file = "data.csv"
        data = process_csv_in_chunks(csv_file)
        if data is None:
            logging.error("Fehler beim Laden der CSV-Datei.")
            return

        if len(data) > 1000:
            logging.info("Datei hat mehr als 1000 Zeilen. Aufteilen in kleinere Dateien...")
            split_csv(csv_file)

            # Verarbeite jede aufgeteilte Datei
            data_dir = "data"
            for filename in os.listdir(data_dir):
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_dir, filename)
                    logging.info(f"Verarbeite Datei: {file_path}")

                    data = process_csv_in_chunks(file_path)
                    if data is None:
                        logging.error("Fehler beim Laden der CSV-Datei.")
                        return

                    categories = data['Kategorie'].unique()
                    category_nodes = initialize_quiz_network(categories)
                    questions = [{"question": row['Frage'], "category": row['Kategorie'], "answer": row['Antwort']} for _, row in data.iterrows()]

                    personality_distributions = {category: random.uniform(0.5, 0.8) for category in [node.label for node in category_nodes]}
                    activation_history, weights_history = simulate_learning(data, category_nodes, personality_distributions)

                    save_model_with_questions_and_answers(category_nodes, questions)
        else:
            logging.info("Datei hat weniger als 1000 Zeilen. Keine Aufteilung erforderlich.")
            categories = data['Kategorie'].unique()
            category_nodes = initialize_quiz_network(categories)
            questions = [{"question": row['Frage'], "category": row['Kategorie'], "answer": row['Antwort']} for _, row in data.iterrows()]

            personality_distributions = {category: random.uniform(0.5, 0.8) for category in [node.label for node in category_nodes]}
            activation_history, weights_history = simulate_learning(data, category_nodes, personality_distributions)

            save_model_with_questions_and_answers(category_nodes, questions)

    end_time = time.time()
    logging.info(f"Simulation abgeschlossen. Gesamtdauer: {end_time - start_time:.2f} Sekunden")
```

### Simulation aus der GUI starten

```python
def run_simulation_from_gui(learning_rate, decay_rate, reward_interval, epochs):
    global model_saved
    model_saved = False  # Erzwinge das Speichern nach dem Training

    start_time = time.time()
    csv_file = "data.csv"

    category_nodes, questions = load_model_with_questions_and_answers("model_with_qa.json")

    if category_nodes is None:
        data = process_csv_in_chunks(csv_file)
        if not isinstance(data, pd.DataFrame):
            logging.error("Fehler beim Laden der CSV-Datei. Erwarteter DataFrame wurde nicht zurückgegeben.")
            return

        if len(data) > 1000:
            logging.info("Datei hat mehr als 1000 Zeilen. Aufteilen in kleinere Dateien...")
            split_csv(csv_file)

            # Verarbeite jede aufgeteilte Datei
            data_dir = "data"
            for filename in os.listdir(data_dir):
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_dir, filename)
                    logging.info(f"Verarbeite Datei: {file_path}")

                    data = process_csv_in_chunks(file_path)
                    if not isinstance(data, pd.DataFrame):
                        logging.error("Fehler beim Laden der CSV-Datei. Erwarteter DataFrame wurde nicht zurückgegeben.")
                        return

                    categories = data['Kategorie'].unique()
                    category_nodes = initialize_quiz_network(categories)
                    questions = [{"question": row['Frage'], "category": row['Kategorie'], "answer": row['Antwort']} for _, row in data.iterrows()]

                    personality_distributions = {category: random.uniform(0.5, 0.8) for category in [node.label for node in category_nodes]}
                    activation_history, weights_history = simulate_learning(
                        data, category_nodes, personality_distributions,
                        epochs=int(epochs),
                        learning_rate=float(learning_rate),
                        reward_interval=int(reward_interval),
                        decay_rate=float(decay_rate)
                    )

                    save_model_with_questions_and_answers(category_nodes, questions)
        else:
            logging.info("Datei hat weniger als 1000 Zeilen. Keine Aufteilung erforderlich.")
            categories = data['Kategorie'].unique()
            category_nodes = initialize_quiz_network(categories)
            questions = [{"question": row['Frage'], "category": row['Kategorie'], "answer": row['Antwort']} for _, row in data.iterrows()]

            personality_distributions = {category: random.uniform(0.5, 0.8) for category in [node.label for node in category_nodes]}
            activation_history, weights_history = simulate_learning(
                data, category_nodes, personality_distributions,
                epochs=int(epochs),
                learning_rate=float(learning_rate),
                reward_interval=int(reward_interval),
                decay_rate=float(decay_rate)
            )

            save_model_with_questions_and_answers(category_nodes, questions)
    else:
        data = process_csv_in_chunks(csv_file)
        if not isinstance(data, pd.DataFrame):
            logging.error("Fehler beim Laden der CSV-Datei. Erwarteter DataFrame wurde nicht zurückgegeben.")
            return

        logging.info(f"Anzahl der Zeilen in der geladenen CSV: {len(data)}")

        personality_distributions = {category: random.uniform(0.5, 0.8) for category in [node.label for node in category_nodes]}

        activation_history, weights_history = simulate_learning(
            data, category_nodes, personality_distributions,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            reward_interval=int(reward_interval),
            decay_rate=float(decay_rate)
        )

        save_model_with_questions_and_answers(category_nodes, questions)

    end_time = time.time()
    logging.info(f"Simulation abgeschlossen. Gesamtdauer: {end_time - start_time:.2f} Sekunden")
    messagebox.showinfo("Ergebnis", f"Simulation abgeschlossen! Dauer: {end_time - start_time:.2f} Sekunden")
```

### Netzwerk asynchron initialisieren

```python
def async_initialize_network():
    global category_nodes, questions, model_saved
    logging.info("Starte Initialisierung des Netzwerks...")

    category_nodes, questions = load_model_with_questions_and_answers("model_with_qa.json")

    if category_nodes is None:
        category_nodes = []
        logging.warning("Keine gespeicherten Kategorien gefunden. Neues Netzwerk wird erstellt.")
        model_saved = False  # Zurücksetzen der Speicher-Flagge

    if questions is None:
        questions = []
        logging.warning("Keine gespeicherten Fragen gefunden. Neues Fragen-Array wird erstellt.")
        model_saved = False  # Zurücksetzen der Speicher-Flagge

    if not category_nodes:
        csv_file = "data.csv"
        data = process_csv_in_chunks(csv_file)
        if isinstance(data, pd.DataFrame):
            if len(data) > 1000:
                logging.info("Datei hat mehr als 1000 Zeilen. Aufteilen in kleinere Dateien...")
                split_csv(csv_file)

                # Verarbeite jede aufgeteilte Datei
                data_dir = "data"
                for filename in os.listdir(data_dir):
                    if filename.endswith(".csv"):
                        file_path = os.path.join(data_dir, filename)
                        logging.info(f"Verarbeite Datei: {file_path}")

                        data = process_csv_in_chunks(file_path)
                        if isinstance(data, pd.DataFrame):
                            categories = data['Kategorie'].unique()
                            category_nodes = initialize_quiz_network(categories)
                            questions = [{"question": row['Frage'], "category": row['Kategorie'], "answer": row['Antwort']} for _, row in data.iterrows()]
                            logging.info("Netzwerk aus CSV-Daten erfolgreich erstellt.")
                            model_saved = False  # Zurücksetzen der Speicher-Flagge
            else:
                logging.info("Datei hat weniger als 1000 Zeilen. Keine Aufteilung erforderlich.")
                categories = data['Kategorie'].unique()
                category_nodes = initialize_quiz_network(categories)
                questions = [{"question": row['Frage'], "category": row['Kategorie'], "answer": row['Antwort']} for _, row in data.iterrows()]
                logging.info("Netzwerk aus CSV-Daten erfolgreich erstellt.")
                model_saved = False  # Zurücksetzen der Speicher-Flagge
        else:
            logging.error("Fehler beim Laden der CSV-Daten. Netzwerk konnte nicht initialisiert werden.")
            return

    save_model_with_questions_and_answers(category_nodes, questions)
    logging.info("Netzwerk erfolgreich initialisiert.")
```

### GUI starten

```python
def start_gui():
    def start_simulation():
        try:
            threading.Thread(target=run_simulation_from_gui, args=(0.8, 0.002, 5, 10), daemon=True).start()
            messagebox.showinfo("Info", "Simulation gestartet!")
            logging.info("Simulation gestartet")
        except Exception as e:
            logging.error(f"Fehler beim Start der Simulation: {e}")
            messagebox.showerror("Fehler", f"Fehler: {e}")

    root = tk.Tk()
    root.title("DRLCogNet GUI")
    root.geometry("400x300")

    header_label = tk.Label(root, text="Simulationseinstellungen", font=("Helvetica", 16))
    header_label.pack(pady=10)

    start_button = tk.Button(root, text="Simulation starten", command=start_simulation)
    start_button.pack(pady=20)

    root.mainloop()
```

### Hauptprogramm

```python
if __name__ == "__main__":
    # Starte die Initialisierung in einem Thread
    threading.Thread(target=async_initialize_network, daemon=True).start()
    start_gui()
```

## Fragen zur Datenbank (SQLite)

### Wird die Datenbank im Arbeitsspeicher erstellt?

Ja, die SQLite-Datenbank wird im Arbeitsspeicher erstellt, wenn die Funktion `save_to_sqlite` aufgerufen wird. Diese Funktion erstellt eine SQLite-Datenbankdatei (standardmäßig `dataset.db`), die im Arbeitsspeicher gespeichert wird, wenn Sie sie nicht an einem anderen Ort speichern.

### Wie wird die Datenbank erstellt?

Die Datenbank wird erstellt, indem eine Verbindung zur SQLite-Datenbank hergestellt wird. Wenn die Datei `dataset.db` nicht existiert, wird sie erstellt. Anschließend werden die Daten aus der CSV-Datei in Chunks gelesen und in die Tabelle `qa_data` der SQLite-Datenbank gespeichert.

### Wie werden die Daten in die Datenbank geladen?

Die Daten werden in Chunks aus der CSV-Datei gelesen und in die Tabelle `qa_data` der SQLite-Datenbank gespeichert. Die Funktion `to_sql` von Pandas wird verwendet, um die Daten in die Datenbank zu schreiben.

### Wie werden die Daten aus der Datenbank geladen?

Die Daten werden aus der Datenbank geladen, indem eine Verbindung zur SQLite-Datenbank hergestellt und eine SQL-Abfrage ausgeführt wird, um die Daten aus der Tabelle `qa_data` zu lesen. Die Funktion `read_sql_query` von Pandas wird verwendet, um die Daten in einen Pandas-DataFrame zu laden.

### Beispielcode zur Verwendung der Datenbank

```python
# Daten in die Datenbank speichern
save_to_sqlite("data.csv")

# Daten aus der Datenbank laden
data = load_from_sqlite()
```

## Fazit

Diese Dokumentation bietet eine umfassende Übersicht über den Code und die Verwendung der SQLite-Datenbank zur Speicherung und zum Laden von Daten. Der Code ist modular aufgebaut und ermöglicht die Verarbeitung und Simulation von Daten aus einer CSV-Datei in einem neuronalen Netzwerk. Die SQLite-Datenbank wird im Arbeitsspeicher erstellt und ermöglicht die effiziente Speicherung und das Laden von Daten.
