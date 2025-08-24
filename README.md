\# RED-GNN Final Visualization



This notebook (`RED\_GNN\_Visual.ipynb`) provides an end-to-end workflow for running knowledge graph inference experiments in \*\*transductive\*\* and \*\*inductive\*\* scenarios, with visualization of predictions and subgraphs.



\## How to Run



Follow the steps below in order:



\### 1. Install Dependencies

\- \*\*Run Cell 1\*\*

&nbsp; Installs the required `consist` library and other dependencies.



\### 2. Mount Google Drive

\- \*\*Run Cell 2\*\*

&nbsp; Mounts your Google Drive to access datasets and model folders.



\### 3. Navigate to Model Folder

\- \*\*Run Cells 3 \& 4\*\*

&nbsp; Modify the paths inside these cells depending on which scenario you want:

&nbsp; - \*\*Transductive scenario\*\* → navigate to the \*\*transductive\*\* folder

&nbsp; - \*\*Inductive scenario\*\* → navigate to the \*\*inductive\*\* folder

&nbsp; These folders contain the models for each scenario.



\### 4. Import Required Libraries

\- \*\*Run Cell 5\*\*

&nbsp; Loads all needed libraries.



\### 5. Implement KGInference

\- \*\*Run Cell 6\*\*

&nbsp; Implements the `KGInference` class for:

&nbsp; - Tracing predictions

&nbsp; - Visualizing results



\### 6. Transductive Scenario

\- \*\*Run Cell 7\*\*

&nbsp; - Modify `data\_path` to choose the dataset.

&nbsp; - You can select one of the two continuous blocks inside the cell:

&nbsp;   - \*\*First block:\*\* Load pretrained best weights

&nbsp;   - \*\*Second block:\*\* Train from scratch



\### 7. Inductive Scenario

\- \*\*Run Cell 8\*\*

&nbsp; Works the same as Cell 7, but for the \*\*inductive\*\* setting.



\### 8. Initialize Visualization

\- \*\*Run Cell 9\*\*

&nbsp; Initializes the `KGInference` class to prepare for visualization.



\### 9. Generate Visualizations

\- \*\*Run Cell 10\*\*

&nbsp; - Modify `head\_name` and `rel\_name` in the `(h, r, ?)` triplet.

&nbsp; - Use:

&nbsp;   - `predict\_tail()` → predict the tail entity

&nbsp;   - `visualize\_subgraph()` → visualize the subgraph

&nbsp; - You can also provide a full `(h, r, t)` triplet using `get\_info()` to inspect details.

&nbsp; - The `alpha` parameter controls the threshold for removing edges/paths during visualization.



---



\## Notes

\- Ensure that your dataset and model weights are correctly placed in your Google Drive before starting.

\- Adjust folder paths in Cells 3 \& 4 carefully to match your setup.

\- For visualization clarity, experiment with different `alpha` values.

