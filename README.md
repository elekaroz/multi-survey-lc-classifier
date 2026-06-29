# Multi-Survey Light Curve Classifier

Pipeline de clasificación de transitorias y otros objetos variables combinando
datos de ZTF, LSST y ATLAS, desarrollado como parte de un Trabajo de Fin de
Máster (TFM) para la Universidad de La Laguna (ULL) y el Instituto de Astrofísica de Canarias (IAC).
Construido sobre el framework `lc_classifier` del broker
[ALeRCE](https://github.com/alercebroker/lc_classifier/tree/main).

El sistema combina **cinco modelos base** (`BalancedRandomForestClassifier`)
con un metamodelo de stacking (`LogisticRegression`):

| Modelo | Survey(s) |
|---|---|
| A | ZTF |
| B | LSST |
| C | ZTF + LSST + ATLAS (features cross-survey) |
| D | Curva de luz combinada (ZTF+LSST) |
| E | ATLAS |

Taxonomía de 15 clases: `SNIa, SNIbc, SNII, SLSN, QSO, AGN, Blazar, YSO,
CV/Nova, RRL, CEP, DSCT, LPV, E, Periodic-Other`.

---

## Tabla de contenidos

- [Instalación](#instalación)
- [Estructura de carpetas](#estructura-de-carpetas)
- [Orden de ejecución](#orden-de-ejecución)
- [Adquisición de datos reales](#adquisición-de-datos-reales)
  - [ztf_lsst.py](#ztf_lsstpy)
  - [atlas.py](#atlaspy)
  - [xmatch.py](#xmatchpy)
- [Simulación](#simulación)
  - [generate.py](#simulationsgeneratepy)
  - [simulate_atlas_only.py](#simulationssimulate_atlas_onlypy)
  - [visualize_atlas_lightcurves.ipynb](#simulationsvisualize_atlas_lightcurvesipynb)
- [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
  - [model_training_full.py](#model_training_fullpy)
  - [feature_selection.py](#feature_selectionpy)
  - [model_training_adapted.py](#model_training_adaptedpy)
  - [bootstrap_confmat.py](#bootstrap_confmatpy)
  - [eval_real_testset.py](#eval_real_testsetpy)
- [Clasificar objetos nuevos: classify.py](#clasificar-objetos-nuevos-classifypy)
  - [Probar con los datos de ejemplo](#probar-con-los-datos-de-ejemplo)
- [Explorar resultados: visualize_results.ipynb](#explorar-resultados-visualize_resultsipynb)
- [Notas](#notas)
- [Atribuciones](#atribuciones)

---

## Instalación

```bash
git clone <url-de-este-repo>
cd repo/
conda env create -f environment.yml
conda activate tfm
```

`environment.yml` instala también el fork modificado de `lc_classifier` incluido en el
repositorio (`lc_classifier/`), en modo editable. No requiere ningún paso aparte. Ha sido modificado
para que funcione con las bandas de LSST y ATLAS.

Algunos modelos entrenados (>50 MB) no van en el repositorio principal por tamaño. Para
poder usar `classify.py` con los 5 modelos base, hay que descargar el
[Release](https://github.com/elekaroz/multi-survey-lc-classifier/releases/tag/v1.0)
y descomprimirlo en `output/models/` (ver detalle en
[Estructura de carpetas](#estructura-de-carpetas)):

```bash
wget https://github.com/elekaroz/multi-survey-lc-classifier/releases/download/v1.0/modelos_pesados.zip
unzip modelos_pesados.zip -d output/models/
```

## Estructura de carpetas

```
repo/
├── environment.yml
├── lc_classifier/              # fork de ALeRCE lc_classifier (instalado vía environment.yml)
│
├── ztf_lsst.py, atlas.py		# query + extracción de curvas de luz reales
├── xmatch.py        			# crossmatch con catálogos para etiquetado
├── model_training_full.py                  # entrenamiento (sin adaptación de dominio)
├── model_training_adapted.py                # entrenamiento adaptado (CORAL + aumentado)
├── feature_selection.py                     # selección de features (consenso SHAP/KS)
├── bootstrap_confmat.py                     # matrices de confusión con límites de confianza
├── eval_real_testset.py                     # evaluación sobre datos reales
├── classify.py                              # clasificación, uso principal en inferencia
├── model_training_functions.py              # librería compartida para aprendizaje (no se ejecuta sola)
├── visualize_results.ipynb                  # explorador de resultados de classify.py
│
├── simulations/                 # generación de la muestra de entrenamiento simulado
│   ├── generate.py, simulate_atlas_only.py	#simulación en 	ZTF/LSST y ATLAS
│   ├── visualize_atlas_lightcurves.ipynb	#explorador de curvas de luz simuladas
│   └── models.py, magnetar_source.py, survey.py, formatter.py,
│       features_config.py, simulate_bandpasses.py   # librerías internas
│
├── examples/                    # datos de ejemplo para probar el clasificador
│   ├── raw_features/             # formato real (det/obj/features de ztf_lsst.py + atlas.py)
│   └── simulated/                # formato simulado (det/obj/features de generate.py)
│
├── data/
│   ├── simulated/        # salida de generate.py / simulate_atlas_only.py
│   ├── raw_features/     # salida de ztf_lsst.py / atlas.py (datos reales sin etiquetar)
│   ├── real/             # salida de xmatch.py: features + etiquetas de la muestra real
│   ├── unlabeled/        # objetos sin etiqueta conocida → entrada de classify.py
│   ├── simlibs/          # cadencia/ruido real (se incluyen ZTF y ATLAS)
│   │   ├── atlas/         # atlas_{c,o}_{cadence,skynoise,zp}.npy, simlib_summary.json
│   │   ├── ztf/            # ztf_fieldlog.parquet
│   │   └── lsst/           # NO incluido, obtener directamente de LSST
│   ├── filter_profiles/  # curvas de transmisión (se incluyen)
│   │   ├── Misc_Atlas.{cyan,orange}.dat.txt      # ATLAS, usadas por simulate_bandpasses.py
│   │   └── LSST_LSST.*.dat.txt, Palomar_ZTF.*.dat.txt  # fallback opcional, no requeridas
│   └── feature_selection/
│
└── output/
    ├── models/      # modelos entrenados, se incluye la versión final del TFM
    ├── coral/       # transformaciones CORAL del entrenamiento adaptado, ídem
    ├── real_eval/   # salida de eval_real_testset.py
    ├── classify/    # salida de classify.py
    └── plots/, figures/, oof/
```

**Ficheros reales de `output/models/`** (estructura jerárquica:
Transient/Stochastic/Periodic, ver `model_training_functions.py`):
- `model_{A,B,C,D,E}_hier.pkl` + `model_{A,B,C,D,E}_{Transient,Stochastic,Periodic}.pkl`
- `calibrators_model{A,B,C,D,E}.pkl`
- `meta_model.pkl`: metamodelo del ensemble **adaptado** (el que usa `classify.py --use-meta` por defecto)
- `meta_model_baseline.pkl`: metamodelo del ensemble **baseline** (`model_training_full.py`), útil para `eval_real_testset.py --compare-metamodels`
- `shap_values_{A,B,C,D,E}.pkl`, `results_summary.xlsx`, `model_comparison_summary.csv`, `test_predictions.pkl`

**Nota sobre ficheros pesados (>50 MB)**: `model_{A,B,C,D,E}_hier.pkl`,
`model_{A,B,C,D,E}_{Transient,Stochastic,Periodic}.pkl` y
`shap_values_C.pkl` no van en el repositorio por tamaño (algunos superan el
límite de 100 MB de GitHub). Se distribuyen como asset de un
[Release](https://github.com/elekaroz/multi-survey-lc-classifier/releases/tag/v1.0):

```bash
wget https://github.com/elekaroz/multi-survey-lc-classifier/releases/download/v1.0/modelos_pesados.zip
unzip modelos_pesados.zip -d output/models/
```

Sin estos ficheros, `classify.py` solo puede usar `--active-models` con
los modelos cuyos `_hier.pkl` sí estén presentes — por defecto los
necesita todos.

**Ficheros reales de `output/coral/`**: `coral_modelB.pkl`, `coral_modelD.pkl`,
y el modelo C partido por grupo de features: `coral_modelC_{ztf,lsst,atlas,diff}.pkl`.

**Nota sobre el OpSim de LSST**: `baseline_v5.0.0_10yrs.db` pesa ~835 MB —
demasiado para un repositorio normal de GitHub, así que **no se
incluye**. Se puede descargar de la
[web oficial de Rubin/OpSim](https://www.lsst.org/scientists/simulations/opsim)
y usar con `--opsim-db /ruta/local/baseline_v5.0.0_10yrs.db` a `generate.py`;
si se omite, se usa cadencia LSST sintética.

Todas las rutas por defecto son relativas a la raíz del repositorio y configurables
por línea de comandos.

## Orden de ejecución (para uso desde cero)

1. **Generar datos de entrenamiento simulados**: `generate.py` →
   (opcional) `simulate_atlas_only.py`
2. **Adquirir y etiquetar datos reales**: `ztf_lsst.py` → `atlas.py` →
   `xmatch.py`
3. **Entrenar**: `model_training_full.py` → `feature_selection.py` →
   `model_training_adapted.py`
4. **Evaluar**: `eval_real_testset.py`, `bootstrap_confmat.py`
5. **Usar el modelo**: `classify.py` sobre objetos nuevos →
   `visualize_results.ipynb` para explorar los resultados

Se incluyen los modelos entrenados en el repositorio, así que en este caso puede usarse `classify.py` directamente.
---

## Adquisición de datos reales

### `ztf_lsst.py`

Descarga y extrae features ZTF/LSST de objetos reales vía la API de ALeRCE.

```bash
python ztf_lsst.py --survey-mode both --oids-source ./mis_oids.csv \
    --output-base ./data/raw_features/
```

| Argumento | Default | Descripción |
|---|---|---|
| `--survey-mode` | `lsst_only` | `both` / `ztf_only` / `lsst_only` |
| `--no-branch-b` | (activada) | Desactiva la rama relaxed (solo con `--survey-mode both`) |
| `--oids-source` | ejemplo en el script | `.csv`, directorio con `ztf_list.txt`/`lsst_list.txt`, o lista de OIDs separados por comas |
| `--output-base` | `./data/raw_features/` | Carpeta de salida (features + checkpoints) |
| `--checkpoint-fetch` / `--checkpoint-extract` | `100` / `100` | Checkpoint cada N objetos |
| `--skip-fetch` | `False` | Omite la descarga, usa checkpoints existentes |
| `--fetch-delay` / `--retry-403-wait` | `0.5` / `60` | Segundos entre requests / espera tras un 403 |

### `atlas.py`

Pide fotometría forzada de ATLAS para los objetos extraídos por
`ztf_lsst.py` y extrae sus features.

```bash
python atlas.py --username U --password P          # descarga + extracción completa
python atlas.py --username U --password P --submit-only
python atlas.py --username U --password P --poll-only
python atlas.py --extract-only                       # solo extraer, sin pedir nada nuevo
```

| Argumento | Default | Descripción |
|---|---|---|
| `--username` / `--password` | *(requeridos salvo `--extract-only`)* | Credenciales de la cuenta ATLAS forced-phot |
| `--obj-glob` | `./data/raw_features/checkpoints_both/obj_*.parquet` | `obj_*.parquet` de `ztf_lsst.py` |
| `--output-dir` | `./data/raw_features/atlas/` | Carpeta de salida (curvas descargadas + checkpoints) |
| `--feat-dir` | `./data/raw_features/` | Features ZTF/LSST (mismo `--output-base` usado en `ztf_lsst.py`)|
| `--pre-buffer` / `--post-buffer` | `30.0` / `30.0` | Días de margen antes/después del rango de detección |
| `--checkpoint-n` | `100` | Checkpoint de features cada N objetos |
| `--submit-only` / `--poll-only` / `--extract-only` | `False` | Ejecutar solo esa fase |

### `xmatch.py`

Construye la muestra de entrenamiento real: cruza los objetos de `ztf_lsst.py`/`atlas.py`
contra TNS, SIMBAD, Milliquas y VSX para asignarles una etiqueta.

```bash
export TNS_API_KEY="..."; export TNS_BOT_ID="..."; export TNS_BOT_NAME="..."
python xmatch.py --features-dir ./data/raw_features/ --output ./data/real/labels_testset.csv
```

| Argumento | Default | Descripción |
|---|---|---|
| `--features-dir` | *requerido* (salvo `--coords-csv`) | Parquets de `ztf_lsst.py`/`atlas.py` |
| `--output` | *requerido* | CSV de labels de salida |
| `--survey-mode` | `both` | `both` / `lsst_only` / `ztf_only` |
| `--obj-dir` | `--features-dir` | Ficheros `obj_*.parquet` (tienen RA/Dec) |
| `--coords-csv` | `None` | Atajo: CSV `oid,ra,dec` directo, sin pasar por parquets |
| `--radius` | `1.5` | Radio de crossmatch (arcsec) |
| `--tns-api-key` / `--tns-bot-id` / `--tns-bot-name` | `$TNS_API_KEY` / `$TNS_BOT_ID` / `$TNS_BOT_NAME` | Credenciales TNS |
| `--skip-simbad` / `--skip-milliquas` / `--skip-vsx` | `False` | Omitir esa fuente |
| `--min-per-class` | `10` | Avisa si una clase tiene menos de N objetos |

---

## Simulación

Generan la muestra entrenamiento simulado en `data/simulated/`. Se ejecutan
desde `simulations/`. `models.py`, `magnetar_source.py`, `survey.py`,
`formatter.py`, `features_config.py` y `simulate_bandpasses.py` son
librerías internas, no se ejecutan directamente.

### `simulations/generate.py`

Motor principal: simula objetos ZTF+LSST de las 15 clases.

```bash
cd simulations/
python generate.py --n-per-class 1000 --out-dir ../data/simulated/ \
    --ztf-fieldlog ../data/simlibs/ztf/ztf_fieldlog.parquet \
    --opsim-db /ruta/a/baseline_v5.0.0_10yrs.db
```

| Argumento | Default | Descripción |
|---|---|---|
| `--n-per-class` | `1000` | Objetos por clase |
| `--out-dir` | `../data/simulated/` | Salida: features + `labels.csv` |
| `--seed` | `42` | Semilla aleatoria |
| `--checkpoint-n` | `100` | Checkpoint cada N objetos por clase |
| `--ztf-fieldlog` | `../data/simlibs/ztf/ztf_fieldlog.parquet` | Cadencia real ZTF; si no existe, sintética |
| `--min-visits-per-year` | `30.0` | Umbral de visitas/año para campo ZTF "bien muestreado" |
| `--opsim-db` | `None` | OpSim de Rubin (cadencia real LSST); si se omite, sintética. **No incluido** (descargar de la [web de Rubin/OpSim](https://www.lsst.org/scientists/simulations/opsim)) |
| `--n-jobs` | `1` | Workers paralelos para extracción de features |

### `simulations/simulate_atlas_only.py`

Añade observaciones ATLAS simuladas a objetos ya simulados en ZTF+LSST (reutiliza el
mismo seed/parámetros físicos).

```bash
python simulate_atlas_only.py \
    --input-dir ../data/simulated/checkpoints --output-dir ../data/simulated/atlas \
    --simlib-dir ../data/simlibs/atlas --bandpass-dir ../data/filter_profiles \
    --n-workers 8 --extract-features
```

| Argumento | Default | Descripción |
|---|---|---|
| `--input-dir` | `../data/simulated/checkpoints/` | Checkpoints de `generate.py` |
| `--output-dir` | `../data/simulated/atlas/` | Salida: detecciones + features ATLAS |
| `--simlib-dir` | `../data/simlibs/atlas/` | SIMLIB ATLAS (incluido) |
| `--bandpass-dir` | `../data/filter_profiles/` | Curvas de transmisión ATLAS c/o (incluidas)|
| `--n-workers` / `--chunk-size` | `4` / `500` | Workers paralelos / objetos por checkpoint (simulaciones)|
| `--n-objects` | `1000` | Target de objetos válidos por clase |
| `--classes` | `None` (→ todas) | Subconjunto de clases a procesar |
| `--extract-features` | `False` | Extrae features ATLAS tras simular |
| `--skip-sim` | `False` | Solo extrae features desde detecciones ya simuladas |
| `--checkpoint-n` | `500` | Checkpoint de features cada N objetos |

### `simulations/visualize_atlas_lightcurves.ipynb`

Notebook para explorar visualmente las curvas de luz simuladas: objetos
aleatorios por clase, comparación de cadencia/profundidad entre surveys,
plegado en fase para clases periódicas. Editar directamente las
celdas de configuración del notebook.

---

## Entrenamiento y evaluación

### `model_training_full.py`

Entrenamiento (baseline, no adaptado): cinco modelos base + metamodelo de stacking sobre
el conjunto simulado, sin selección de features ni adaptación de dominio. Cuando exista
una muestra real de entrenamiento, usar este programa para crear los nuevos modelos.

```bash
python model_training_full.py --output-dir ./output/ --meta-dropout-prob 0.3
```

| Argumento | Default | Descripción |
|---|---|---|
| `--labels-file` | `./data/simulated/labels.csv` | Etiquetas de la muestra |
| `--features-ztf` / `--features-lsst` / `--features-atlas` / `--features-combined` | `./data/simulated/features_*.parquet` | Features simuladas |
| `--output-dir` | `./output/` | Carpeta de salida (`models/`, `plots/`, `oof/`) |
| `--meta-dropout-prob` | `0.3` | Probabilidad de dropout por survey en el metamodelo (`0.0` para desactivar) |

### `feature_selection.py`

Decide qué features usar en cada modelo adaptado, comparando importancia
SHAP contra el domain gap de distribución simulado-real (estadístico KS).

```bash
python feature_selection.py --shap-dir ./output/models/ --sim-dir ./data/simulated/ \
    --real-dir ./data/real/ --out-dir ./data/feature_selection/
```

| Argumento | Default | Descripción |
|---|---|---|
| `--shap-dir` | `./output/models/` | `shap_values_{B,C,D}.pkl` |
| `--sim-dir` | `./data/simulated/` | Features simuladas ZTF/LSST |
| `--sim-atlas-dir` | `None` (→ `--sim-dir`) | Features simuladas ATLAS |
| `--real-dir` | `./data/real/` | Features reales *strict* |
| `--real-atlas-dir` | `None` (→ `--real-dir`) | Features reales *strict* ATLAS |
| `--out-dir` | `./data/feature_selection/` | Salida: `consensus_features.csv` |

### `model_training_adapted.py`

Entrenamiento adaptado: selección de features, adaptación CORAL (B/C/D),
dropout de surveys, y aumentation con objetos reales etiquetados.

```bash
python model_training_adapted.py --output-dir ./output/ \
    --consensus-csv ./data/feature_selection/consensus_features.csv
```

| Argumento | Default | Descripción |
|---|---|---|
| `--consensus-csv` | `./data/feature_selection/consensus_features.csv` | Salida de `feature_selection.py` |
| `--features-ztf` / `--features-lsst` / `--features-atlas` / `--features-combined` | `./data/simulated/features_*.parquet` | Features simuladas |
| `--labels-file` | `./data/simulated/labels.csv` | Etiquetas de la muestra simulada |
| `--real-ztf-strict` / `--real-lsst-strict` / `--real-comb-strict` / `--real-atlas-strict` | `./data/real/features_*_strict.parquet` | Target domain de CORAL |
| `--real-labels-file` | `./data/real/labels_testset.csv` | Etiquetas de la muestra real |
| `--output-dir` | `./output/` | Carpeta de salida |
| `--coral-lambda` | `0.1` | Regularización de CORAL |
| `--coral-ks-threshold` | `0.4` | Umbral KS mínimo para adaptar una feature |
| `--include-unknown` | `False` | Conserva features UNKNOWN con SHAP alto |
| `--unknown-shap-percentile` | `75` | Percentil SHAP usado como umbral para UNKNOWN |
| `--no-real-meta-aug` | (activado) | Desactiva el aumentado del metamodelo con objetos reales |
| `--real-meta-upweight` | `10` | Peso de los objetos reales en el metamodelo |
| `--meta-dropout-prob` | `0.3` | Probabilidad de dropout por survey |

### `bootstrap_confmat.py`

Genera matrices de confusión con intervalos de confianza por bootstrap, para el test
simulado y/o real, a partir de modelos ya entrenados.

```bash
python bootstrap_confmat.py --output-dir ./output/ --n-bootstrap 1000
```

| Argumento | Default | Descripción |
|---|---|---|
| `--no-run-sim` / `--no-run-real` | (ambos activados) | No regenerar las matrices del test simulado / real |
| `--output-dir` | `./output/` | Salida de `model_training_adapted.py` |
| `--consensus-csv` | `./data/feature_selection/consensus_features.csv` | CSV de consenso |
| `--eval-output-dir` | `./output/real_eval/` | Salida de `eval_real_testset.py` |
| `--real-labels-file` | `./data/real/labels_testset.csv` | Etiquetas de la muestra real |
| `--real-branch` | `strict` | `strict` o `relaxed` |
| `--n-bootstrap` | `1000` | Remuestreos bootstrap |
| `--random-state` | `42` | Semilla aleatoria |

### `eval_real_testset.py`

Evalúa el clasificador entrenado sobre la muestra de prueba real etiquetada por
`xmatch.py`. Genera matrices de confusión, diagramas de fiabilidad y un
CSV resumen.

```bash
python eval_real_testset.py --features-dir ./data/real/ --models ./output/models/ \
    --output ./output/real_eval/
# Modo adaptado:
python eval_real_testset.py ... --adapted --consensus-csv ... --coral-dir ./output/coral/
```

| Argumento | Default | Descripción |
|---|---|---|
| `--features-dir` | `./data/real/` | `features_{ztf,lsst,comb}_<branch>.parquet` |
| `--features-atlas-dir` | `./data/real/` | `features_atlas_<branch>.parquet` (se omite con aviso si no existe) |
| `--labels` | `./data/real/labels_testset.csv` | Etiquetas de la muestra real |
| `--models` | `./output/models/` | Modelos `.pkl` entrenados |
| `--output` | `./output/real_eval/` | Carpeta de salida |
| `--branches` | `[strict, relaxed]` | Ramas a evaluar |
| `--bootstrap-n` | `1000` | Remuestreos bootstrap |
| `--sim-summary` | `./output/models/results_summary.xlsx` | Excel del modelo baseline, para comparación |
| `--adapted` | `False` | Evalúa modelos adaptados (CORAL + selección) |
| `--consensus-csv` | `None` | CSV del consenso de features. Requerido si `--adapted` |
| `--coral-dir` | `None` | `coral_model*.pkl` |
| `--compare-with` | `None` | `results_summary_real.xlsx` de otro run, para comparar |
| `--metamodel` / `--compare-metamodels` | `None` (→ `meta_model.pkl`) / `None` | Metamodelo a cargar / a comparar |

---

## Clasificar objetos nuevos: `classify.py`

Clasifica objetos **sin etiqueta** con el clasificador ya entrenado. Cada
objeto se clasifica con el subconjunto de modelos base para el que hay
features disponibles (ZTF, LSST, ATLAS, combinada). Los modelos sin datos
disponibles aportan un prior uniforme al metamodelo en vez de excluirse.

### Probar con los datos de ejemplo

El repositorio ya trae el clasificador adaptado entrenado (`output/models/`,
`output/coral/`, salvo los `.pkl` pesados, ver
[Instalación](#instalación)) y un puñado de objetos de ejemplo en
`examples/`. Tras descargar el Release de modelos, se puede probar
`classify.py` sin generar ni descargar datos nuevos:

```bash
python classify.py --models-dir ./output/models/ --coral-dir ./output/coral/ \
    --ztf ./examples/raw_features/features_ztf_strict.parquet \
    --lsst ./examples/raw_features/features_lsst_strict.parquet \
    --use-meta --output ./output/classify/example_results.parquet
```

Y luego abrir `visualize_results.ipynb` con `RESULTS_PATH =
'./output/classify/example_results.parquet'` para explorar el resultado
(ver [Explorar resultados](#explorar-resultados-visualize_resultsipynb)).
`examples/simulated/` trae la muestra de entrenamiento simulada.

**Flujo típico (con datos propios):**
1. Tener ya entrenado el clasificador (`model_training_full.py` o
   `model_training_adapted.py`) → modelos en `./output/models/`.
2. Tener features de los objetos a clasificar en `./data/unlabeled/`
   (mismo formato que produce `ztf_lsst.py`/`atlas.py`).
3. Ejecutar `classify.py` apuntando a esos parquets.
4. Abrir `visualize_results.ipynb` para explorar el resultado, o directamente el CSV de salida.

**Uso (terminal):**
```bash
python classify.py --models-dir ./output/models/ \
    --ztf ./data/unlabeled/features_ztf_relaxed.parquet \
    --lsst ./data/unlabeled/features_lsst_relaxed.parquet \
    --use-meta --output ./output/classify/classify_results.parquet
```

**Uso (Spyder/IPython):** editar el bloque `DIRECT_RUN` al principio del
script y ejecutar con `%runfile`.

| Argumento | Default | Descripción |
|---|---|---|
| `--models-dir` | *requerido* | Carpeta con `model_{A..E}_hier.pkl` + `model_{A..E}_{Transient,Stochastic,Periodic}.pkl`, `calibrators_model{A..E}.pkl`, `meta_model.pkl` |
| `--ztf` / `--lsst` / `--atlas` / `--combined` | `None` | Parquets de features, se omiten los que no estén disponibles |
| `--active-models` | `[A,B,C,D,E]` | Modelos base a usar |
| `--coral-dir` | `None` | `coral_modelB.pkl`, `coral_modelD.pkl`, `coral_modelC_{ztf,lsst,atlas,diff}.pkl` de `model_training_adapted.py` (omitir si los modelos no son adaptados) |
| `--use-meta` | `False` (→ promedio simple) | Usar el metamodelo o el promedio de los modelos base como resultado final |
| `--output` | *requerido* | Ruta de salida (genera `.parquet` y `.csv`) |


El resultado (`classify_results.parquet`) trae, por objeto, la probabilidad
final de cada una de las 15 clases, la clase más probable (la predicción), y las
predicciones individuales de cada modelo base usado.

---

## Explorar resultados: `visualize_results.ipynb`

Notebook interactivo (con widgets) para revisar lo que produjo
`classify.py`, sin necesidad de escribir código.

- **Panel por objeto**: buscar un `oid` concreto y ver sus probabilidades
  finales, el consenso entre modelos base, y un resumen que avisa
  si el metamodelo y los modelos base discrepan demasiado (posible señal de
  un caso atípico, como ocurre con algunos SLSN reclasificados como QSO).
- **Panel por clase**: tabla ordenable y exportable con todos los objetos
  clasificados en una clase dada (útil para revisar candidatos de una
  clase concreta, p. ej. todos los `SLSN` con probabilidad alta).

**Para usarlo**, abrir el notebook y, en la primera celda de
configuración, ajustar:

| Variable | Default | Descripción |
|---|---|---|
| `RESULTS_PATH` | `./output/classify/classify_results.parquet` | Salida de `classify.py` a explorar |
| `SAVES_PATH` | `./output/figures/` | Carpeta donde se guardan las figuras/tablas exportadas |
| `ACTIVE_MODELS` | `['A','B','C','D','E']` | Debe coincidir con `--active-models` usado en `classify.py` |
| `USE_META` | `True` | Debe coincidir con el uso de `--use-meta` en `classify.py` |
| `DIV_THRESHOLD` / `BASE_MAX` | `0.3` / `0.1` | Umbrales para el aviso de divergencia metamodelo/modelos base |

Tras ajustar la celda de configuración, ejecutar todas las celdas
(`Run All`); los paneles son interactivos (desplegables y tablas), no
requieren tocar más código.

---

## Notas

- **Credenciales**: `xmatch.py` necesita una API key de TNS, leída de
  variables de entorno (`TNS_API_KEY`, `TNS_BOT_ID`, `TNS_BOT_NAME`).
  `atlas.py` necesita usuario/contraseña de
  [fallingstar-data.com](https://fallingstar-data.com/forcedphot/), pasados
  por `--username`/`--password`.
- **Convención strict/relaxed**: las features reales (`data/real/`,
  `data/unlabeled/`) vienen en dos ramas. *Strict* exige que cada survey
  pase su propio preprocesador por separado; *relaxed* solo exige que la
  curva combinada lo pase, útil para clasificación temprana con pocas
  detecciones, a costa de features individuales por survey menos fiables.
 
---

## Atribuciones

Este proyecto se construye sobre [`lc_classifier`](https://github.com/alercebroker/lc_classifier)
de ALeRCE (MIT License, © 2023 ALeRCE). Se modificaron sus preprocesadores
y extractores de features (`LSSTLightcurvePreprocessor`,
`ATLASLightcurvePreprocessor`, `LSSTFeatureExtractor`,
`ATLASFeatureExtractor`, entre otros) para soportar las bandas de LSST y
ATLAS, ausentes en el paquete original orientado a ZTF. Su módulo
`features/turbofats/` está a su vez basado en
[FATS](https://github.com/isadoranun/FATS) y
[FATS-2.0](https://github.com/jonwihl/FATS-2.0) (ambos MIT).
