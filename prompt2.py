MERGE_NODE_BY_TRAJECTORY_MOBILE_PROMPT = """You are an assistant skilled in GUI operations. In a GUI environment, actions are taken according to instructions, and each action transitions from the current page to a new page.

I will provide you with a single task trajectory, including the pages visited during the process and the actions taken. I need you to **abstract this trajectory into a state transition sequence**.

---

**Input Format:**

```
Trajectory: <Goal>
- Page: <Page Description>
  Action: <Action Taken>
- Page: <Next Page Description>
  Action: <Action Taken>
- ...
- Page: <Final Page Description>
```

---

**Output Format:**

Please provide the abstracted state transition sequence in the following format:

### State Transition Sequence

```
<Abstract State 1> --[Abstract Action 1]--> <Abstract State 2> --[Abstract Action 2]--> <Abstract State 3> ...
```

### State & Action Descriptions

| Step | Abstract State | Abstract Action | Next Abstract State |
|------|---------------|-----------------|---------------------|
| 1 | S1: ... | A1: ... | S2: ... |
| 2 | S2: ... | A2: ... | S3: ... |
| ... | | | |

---

**Key Principles:**
- **States** should reflect the **functional role** of the page, not its specific content (e.g., "Product Detail Page" rather than "iPhone 15 Detail Page")
- **Actions** should be described **generically** (e.g., "Select item" rather than "Click iPhone 15")
- Focus on capturing the **essential navigation logic** of the task
"""




MERGE_TRAJECTORY_MOBILE_PROMPT = """You are an assistant skilled in GUI operations. In a GUI environment, actions are taken according to instructions, and each action transitions from the current page to a new page. When performing different tasks in an app, similar pages may be encountered; these pages may display different content depending on the task, but the available operations on these pages remain the same.

I will provide you with several different task trajectories, including the pages visited during the process and the actions taken. I need you to:

1. **Analyze** the provided trajectories (page changes and actions between them)
2. **Identify** similar/equivalent pages across different tasks (pages that serve the same functional role even if displaying different content)
3. **Abstract** all trajectories into a unified **state transition graph**, where:
   - **Nodes** = abstract page states (merged similar pages)
   - **Edges** = actions that trigger transitions between states

---

**Input Format:**

```
Trajectory Task n: <Goal>
S1: <Page> --> A1: <Action> --> S2: <Page> --> A2: <Action> --> ... --> Sn: <Final Page>
```

---

**Output Format:**

Please provide:

### 1. Page Abstraction
List the abstract page states identified, explaining which concrete pages from the trajectories map to each abstract state.

| Abstract State | Concrete Pages (from trajectories) | Description |
|---|---|---|
| S1: ... | Traj1-Page1, Traj2-Page1 | ... |

### 2. State Transition Graph
Describe the edges in the format:

```
<Source State> --[Action]--> <Target State>
```

### 3. Summary Diagram (Text-based)
Provide a concise visual representation of the full state transition graph.

---

**Key Principles:**
- Pages with the **same functional role** should be merged into one abstract state, even if their content differs
- Actions on edges should be described **generically** (e.g., "click product" rather than "click iPhone 15")
- If a page can be reached via **multiple actions**, represent all corresponding edges
"""


MAP_SCREEN_TO_STATES_MOBILE_PROMPT = """You are an assistant skilled in GUI operations. In GUI environments, actions are performed according to instructions, and each action transitions the interface from the current screen to a new one. When executing different tasks in a application, you often pass through similar screens; these screens may display different content depending on the task, but the available operations are often the same.

I will provide you with:
1. **A task trajectory** consisting of a sequence of states and actions
2. **An action history** recording the actions that have been taken so far in the trajectory
3. **A single screen's text description** corresponding to one specific state in the trajectory
4. **A next execution action** that will be performed on the current screen
Your task is to **map this screen to the most appropriate state** in the state transition trajectory based on its text content and functional role.

---

**Input Format:**

### State Description

```
State A: <State description>
... 
```

### Task Trajectory

```
S_A --[Action 1]--> S_B --[Action 2]--> S_C --[Action 3]--> ...
```

### Action History

```
Step 1: <Action taken>
Step 2: <Action taken>
Step 3: <Action taken>
...
```

### Current Action

```
<text content of the screen>
```

### Next Execution Action

```
<action that will be performed on the current screen>
```

### Screen Description

```
<image of the screen>
```

---

**Output Format:**

Please output the result in the following JSON format:

```json
{
  "mapped_state": "<State descriptive name>",
  "reason": "..."
}
```

---

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `mapped_state` | `string` | The most appropriate State ID from the graph, or a new descriptive name if no match exists |
| `reason` | `string` | Brief explanation of why this screen was mapped to this state |

---

**Key Principles:**
- Use the **action history** to determine how far along the trajectory the current screen is
- Use the **next execution action** to verify the mapping — the mapped state should have an outgoing edge in the trajectory that matches or is functionally similar to the next action
- Use the **trajectory context** to understand the functional role of the current state (e.g., what actions came before and after)
- Match the screen to the state whose **position in the trajectory** best aligns with the completed actions, current screen content, and next action
- Focus on the **functional purpose** of the screen rather than its specific content details
- Output **only the JSON result**, no additional explanation needed

---
"""







MERGE_NODE_BY_TRAJECTORY_WEB_PROMPT = """You are an assistant skilled in GUI operations. In a GUI environment, actions are taken according to instructions, and each action transitions from the current page to a new page.

I will provide you with a single task trajectory, including the pages visited during the process and the actions taken. I need you to **abstract this trajectory into a state transition sequence**.

---

**Input Format:**

```
Trajectory: <Goal>
- Page: <Page Description>
  Action: <Action Taken>
- Page: <Next Page Description>
  Action: <Action Taken>
- ...
- Page: <Final Page Description>
```

---

**Output Format:**

Please provide the abstracted state transition sequence in the following format:

### State Transition Sequence

```
<Abstract State 1> --[Abstract Action 1]--> <Abstract State 2> --[Abstract Action 2]--> <Abstract State 3> ...
```

### State & Action Descriptions

| Step | Abstract State | Abstract Action | Next Abstract State |
|------|---------------|-----------------|---------------------|
| 1 | S1: ... | A1: ... | S2: ... |
| 2 | S2: ... | A2: ... | S3: ... |
| ... | | | |

---

**Key Principles:**
- **States** should reflect the **functional role** of the page, not its specific content (e.g., "Product Detail Page" rather than "iPhone 15 Detail Page")
- **Actions** should be described **generically** (e.g., "Select item" rather than "Click iPhone 15")
- Focus on capturing the **essential navigation logic** of the task
"""


MERGE_TRAJECTORY_WEB_PROMPT = """You are an assistant skilled in GUI operations. In a GUI environment, actions are taken according to instructions, and each action transitions from the current page to a new page. When performing different tasks in a web, similar pages may be encountered; these pages may display different content depending on the task, but the available operations on these pages remain the same.

I will provide you with several different task trajectories, including the pages visited during the process and the actions taken. I need you to:

1. **Analyze** the provided trajectories (page changes and actions between them)
2. **Identify** similar/equivalent pages across different tasks (pages that serve the same functional role even if displaying different content)
3. **Abstract** all trajectories into a unified **state transition graph**, where:
   - **Nodes** = abstract page states (merged similar pages)
   - **Edges** = actions that trigger transitions between states

---

**Input Format:**

```
Trajectory Task n: <Goal>
S1: <Page> --> A1: <Action> --> S2: <Page> --> A2: <Action> --> ... --> Sn: <Final Page>
```

---

**Output Format:**

Please provide:

### 1. Page Abstraction
List the abstract page states identified, explaining which concrete pages from the trajectories map to each abstract state.

| Abstract State | Concrete Pages (from trajectories) | Description |
|---|---|---|
| S1: ... | Traj1-Page1, Traj2-Page1 | ... |

### 2. State Transition Graph
Describe the edges in the format:

```
<Source State> --[Action]--> <Target State>
```

### 3. Summary Diagram (Text-based)
Provide a concise visual representation of the full state transition graph.

---

**Key Principles:**
- Pages with the **same functional role** should be merged into one abstract state, even if their content differs
- Actions on edges should be described **generically** (e.g., "click product" rather than "click iPhone 15")
- If a page can be reached via **multiple actions**, represent all corresponding edges
"""


TRAJECTORY_TO_STATE_TRANSITION_BY_GRAPH_WEB_PROMPT = """You are an assistant skilled in GUI operations. In GUI environments, actions are performed according to instructions, and each action transitions the interface from the current screen to a new one. When executing different tasks in a web, you often pass through similar screens; these screens may display different content depending on the task, but the available operations are often the same.

We have merged the state transition information from related trajectories into a corresponding state transition diagram. I will provide you with:
1. **The state transition graph** (nodes and edges)
2. **A single task trajectory** to be re-mapped

Your task is to **re-label each state and action** in the trajectory using the abstract nodes and edges from the state transition graph.

---

**Input Format:**

### State Transition Graph

**Nodes:**
| Node ID | Description |
|---------|-------------|
| S_Home | The starting point of the web |
| S_PregnancyTools | Pages related to pregnancy tools |
| ... | ... |

**Edges:**
```
S_Home --[Navigate to Pregnancy Section]--> S_PregnancyTools
S_PregnancyTools --[Select Calculator]--> S_CalculatorInput
...
```

---

### Trajectory to Re-map

```
S1: <Page Description> --> A1: <Action> --> S2: <Page Description> --> A2: <Action> --> ... --> Sn: <Page Description>
```

---

**Output Format:**

Please output the re-mapped trajectory in the following format:

### Re-mapped Trajectory

```
<Node ID> --[<Edge>]--> <Node ID> --[<Edge>]--> ... --> <Node ID>
```

### Mapping Details

| Original | Mapped | Reason |
|----------|--------|--------|
| S1: Home Page | S_Home | Home Page serves as the web entry point |
| A1: Navigate to Pregnancy Section | Navigate to Pregnancy Section | Direct match to graph edge |
| S2: Pregnancy Tools Page | S_PregnancyTools | Pregnancy Tools Page matches the pregnancy tools node |
| ... | ... | ... |

---

**Key Principles:**
- Map each concrete page to the **most functionally similar** node in the graph
- Map each action to the **closest matching edge** in the graph
- If no exact match exists, choose the **nearest equivalent** and explain your reasoning in the "Reason" column
- Maintain the **sequential order** of the original trajectory

---
"""

MAP_SCREEN_TO_STATES_WEB_PROMPT = """You are an assistant skilled in GUI operations. In GUI environments, actions are performed according to instructions, and each action transitions the interface from the current screen to a new one. When executing different tasks in a web, you often pass through similar screens; these screens may display different content depending on the task, but the available operations are often the same.

I will provide you with:
1. **A task trajectory** consisting of a sequence of states and actions
2. **An action history** recording the actions that have been taken so far in the trajectory
3. **A single screen's text description** corresponding to one specific state in the trajectory
4. **A next execution action** that will be performed on the current screen
Your task is to **map this screen to the most appropriate state** in the state transition trajectory based on its text content and functional role.

---

**Input Format:**

### State Description

```
State A: <State description>
... 
```

### Task Trajectory

```
S_A --[Action 1]--> S_B --[Action 2]--> S_C --[Action 3]--> ...
```

### Action History

```
Step 1: <Action taken>
Step 2: <Action taken>
Step 3: <Action taken>
...
```

### Screen Description

```
<text content of the screen>
```

### Current Action

```
<text content of the screen>
```

### Next Execution Action

```
<action that will be performed on the current screen>
```

---

**Output Format:**

Please output the result in the following JSON format:

```json
{
  "mapped_state": "<State descriptive name>",
  "reason": "..."
}
```

---

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `mapped_state` | `string` | The most appropriate State ID from the graph, or a new descriptive name if no match exists |
| `reason` | `string` | Brief explanation of why this screen was mapped to this state |

---

**Key Principles:**
- Use the **action history** to determine how far along the trajectory the current screen is
- Use the **next execution action** to verify the mapping — the mapped state should have an outgoing edge in the trajectory that matches or is functionally similar to the next action
- Use the **trajectory context** to understand the functional role of the current state (e.g., what actions came before and after)
- Match the screen to the state whose **position in the trajectory** best aligns with the completed actions, current screen content, and next action
- Focus on the **functional purpose** of the screen rather than its specific content details
- Output **only the JSON result**, no additional explanation needed

---
"""



TRAJECTORY_TO_STATE_TRANSITION_BY_GRAPH_MOBILE_PROMPT = """You are an assistant skilled in GUI operations. In GUI environments, actions are performed according to instructions, and each action transitions the interface from the current screen to a new one. When executing different tasks in a appliaction, you often pass through similar screens; these screens may display different content depending on the task, but the available operations are often the same.

We have merged the state transition information from related trajectories into a corresponding state transition diagram. I will provide you with:
1. **The state transition graph** (nodes and edges)
2. **A single task trajectory** to be re-mapped

Your task is to **re-label each state and action** in the trajectory using the abstract nodes and edges from the state transition graph.

---

**Input Format:**

### State Transition Graph

**Nodes:**
| Node ID | Description |
|---------|-------------|
| S_Home | The starting point of the web |
| S_PregnancyTools | Pages related to pregnancy tools |
| ... | ... |

**Edges:**
```
S_Home --[Navigate to Pregnancy Section]--> S_PregnancyTools
S_PregnancyTools --[Select Calculator]--> S_CalculatorInput
...
```

---

### Trajectory to Re-map

```
S1: <Page Description> --> A1: <Action> --> S2: <Page Description> --> A2: <Action> --> ... --> Sn: <Page Description>
```

---

**Output Format:**

Please output the re-mapped trajectory in the following format:

### Re-mapped Trajectory

```
<Node ID> --[<Edge>]--> <Node ID> --[<Edge>]--> ... --> <Node ID>
```

### Mapping Details

| Original | Mapped | Reason |
|----------|--------|--------|
| S1: Home Page | S_Home | Home Page serves as the web entry point |
| A1: Navigate to Pregnancy Section | Navigate to Pregnancy Section | Direct match to graph edge |
| S2: Pregnancy Tools Page | S_PregnancyTools | Pregnancy Tools Page matches the pregnancy tools node |
| ... | ... | ... |

---

**Key Principles:**
- Map each concrete page to the **most functionally similar** node in the graph
- Map each action to the **closest matching edge** in the graph
- If no exact match exists, choose the **nearest equivalent** and explain your reasoning in the "Reason" column
- Maintain the **sequential order** of the original trajectory

---
"""