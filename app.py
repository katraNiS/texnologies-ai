import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(page_title="Data Science & ML Web App", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        padding: 0.65rem;
        border-radius: 0.6rem;
        border: 1px solid #e8edf5;
    }
    [data-testid="stHorizontalBlock"] > div:has(> [data-testid="stMetric"]) {
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Apply missing-value strategy on a copy of dataframe."""
    out = df.copy()
    if strategy == "Drop rows with NaN":
        return out.dropna()

    if strategy in [
        "Fill numeric with mean + categorical with mode",
        "Fill numeric with median + categorical with mode",
    ]:
        numeric_cols = out.select_dtypes(include="number").columns.tolist()
        categorical_cols = out.select_dtypes(exclude="number").columns.tolist()

        for col in numeric_cols:
            if strategy.startswith("Fill numeric with mean"):
                out[col] = out[col].fillna(out[col].mean())
            else:
                out[col] = out[col].fillna(out[col].median())

        for col in categorical_cols:
            mode_value = out[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            out[col] = out[col].fillna(fill_value)

    return out


def infer_task_type(y: pd.Series) -> str:
    """Heuristic task type inference from target column."""
    if y.dtype == "object" or str(y.dtype).startswith("category") or y.dtype == "bool":
        return "Classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) <= 20:
        return "Classification"
    return "Regression"


def get_player_per_game_defaults(columns: list[str], goal_mode: str) -> tuple[str, list[str]]:
    """Suggested default setup for Player Per Game dataset."""
    if goal_mode == "Predict player stats (regression)":
        default_target = "pts_per_game" if "pts_per_game" in columns else columns[-1]
        preferred_features = [
            "age",
            "g",
            "gs",
            "mp_per_game",
            "fg_percent",
            "x3p_percent",
            "x2p_percent",
            "e_fg_percent",
            "ft_percent",
            "orb_per_game",
            "drb_per_game",
            "trb_per_game",
            "ast_per_game",
            "stl_per_game",
            "blk_per_game",
            "tov_per_game",
            "pf_per_game",
        ]
    else:
        default_target = "pos" if "pos" in columns else columns[-1]
        preferred_features = [
            "age",
            "g",
            "gs",
            "mp_per_game",
            "fg_per_game",
            "fga_per_game",
            "fg_percent",
            "x3p_per_game",
            "x3pa_per_game",
            "x3p_percent",
            "x2p_per_game",
            "x2pa_per_game",
            "x2p_percent",
            "e_fg_percent",
            "ft_per_game",
            "fta_per_game",
            "ft_percent",
            "orb_per_game",
            "drb_per_game",
            "trb_per_game",
            "ast_per_game",
            "stl_per_game",
            "blk_per_game",
            "tov_per_game",
            "pf_per_game",
            "pts_per_game",
        ]
    defaults = [c for c in preferred_features if c in columns and c != default_target]
    if defaults:
        return default_target, defaults

    fallback = [c for c in columns if c != default_target and c not in {"player", "player_id"}]
    return default_target, fallback


def plot_comment(text: str) -> None:
    """Short interpretation note shown under each chart."""
    st.caption(f"How to read this: {text}")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to UTF-8 CSV bytes for download buttons."""
    return df.to_csv(index=False).encode("utf-8")


def style_fig(fig, height: int = 430):
    """Apply consistent visual style to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


st.title("Project 1: Data Science & Machine Learning Web Application")
st.caption(
    "Pipeline aligned with assignment stages: Data Loading/Preprocessing, EDA, and ML Pipeline."
)

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
all_columns = df_raw.columns.tolist()

if len(all_columns) < 2:
    st.error("Dataset must contain at least 2 columns.")
    st.stop()

is_player_per_game_schema = {
    "player",
    "player_id",
    "season",
    "pos",
    "pts_per_game",
    "ast_per_game",
    "trb_per_game",
}.issubset(set(all_columns))

st.sidebar.subheader("Project Setup")
if is_player_per_game_schema:
    goal_mode = st.sidebar.radio(
        "Goal",
        ["Predict player stats (regression)", "Predict player position (classification)"],
        index=0,
        help="Choose the prediction objective for your report/demo. This affects default target and model type.",
    )
else:
    goal_mode = "Auto"

active_seasons = None
if "season" in all_columns and pd.api.types.is_numeric_dtype(df_raw["season"]):
    unique_seasons = sorted(df_raw["season"].dropna().astype(int).unique().tolist())
    default_recent_only = is_player_per_game_schema
    recent_only = st.sidebar.checkbox(
        "Use only most recent seasons",
        value=default_recent_only,
        help="Recommended for the project narrative: focus analysis on the latest player behavior.",
    )
    if recent_only and unique_seasons:
        max_window = min(10, len(unique_seasons))
        window = st.sidebar.slider(
            "Number of recent seasons",
            1,
            max_window,
            min(2, max_window),
            help="Set to 2 if you want exactly a two-season analysis.",
        )
        active_seasons = unique_seasons[-window:]
        df_raw = df_raw[df_raw["season"].isin(active_seasons)].copy()
        st.sidebar.caption(f"Using seasons: {active_seasons}")
        all_columns = df_raw.columns.tolist()

if df_raw.empty:
    st.error("No rows available after the selected season filter. Change sidebar settings.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Current Dataset")
st.sidebar.caption(f"File: `{uploaded_file.name}`")
st.sidebar.caption(f"Rows: {df_raw.shape[0]} | Columns: {df_raw.shape[1]}")
if active_seasons is not None:
    st.sidebar.caption(f"Seasons: {active_seasons}")
if is_player_per_game_schema:
    st.sidebar.caption(f"Goal: {goal_mode}")

tabs = st.tabs(["1. Preprocessing", "2. EDA", "3. ML Pipeline"])

with tabs[0]:
    st.subheader("Load Data and Preprocess")
    with st.expander("What to do in this stage", expanded=False):
        st.markdown(
            """
            1. Select your `target` (what you want to predict).
            2. Select `features` (inputs to the model).
            3. Choose missing-value, scaling, and encoding options.
            4. Confirm the processed preview before moving to EDA.
            """
        )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", int(df_raw.shape[0]))
    col_b.metric("Columns", int(df_raw.shape[1]))
    col_c.metric("Total Missing Values", int(df_raw.isna().sum().sum()))

    preview_rows = st.slider(
        "Preview rows",
        min_value=5,
        max_value=min(50, max(5, int(df_raw.shape[0]))),
        value=min(10, max(5, int(df_raw.shape[0]))),
        step=5,
        help="Adjust how many rows you want to preview.",
    )
    data_col1, data_col2 = st.columns([1, 2])
    with data_col1:
        st.write("Data types")
        st.dataframe(df_raw.dtypes.astype(str).rename("dtype"), use_container_width=True, height=320)
    with data_col2:
        st.write("First rows")
        st.dataframe(df_raw.head(preview_rows), use_container_width=True, height=320)
    if active_seasons is not None:
        st.caption(f"Dataset filtered to recent seasons: {active_seasons}")

    target_default_name, features_default_names = get_player_per_game_defaults(all_columns, goal_mode)
    target_default_idx = all_columns.index(target_default_name) if target_default_name in all_columns else len(all_columns) - 1
    if is_player_per_game_schema and goal_mode == "Predict player stats (regression)":
        st.info("Detected Player Per Game schema. Suggested setup: recent seasons + regression target `pts_per_game`.")
    elif is_player_per_game_schema:
        st.info("Detected Player Per Game schema. Suggested target: `pos`.")

    target_col = st.selectbox(
        "Select target variable",
        options=all_columns,
        index=target_default_idx,
        help="The model will try to predict this column.",
    )

    default_features = [c for c in features_default_names if c != target_col]
    if not default_features:
        default_features = [c for c in all_columns if c != target_col]
    selected_features = st.multiselect(
        "Select feature variables",
        options=[c for c in all_columns if c != target_col],
        default=default_features,
        help="These columns are used as model inputs. Exclude IDs or pure text fields unless needed.",
    )

    if not selected_features:
        st.warning("Select at least one feature to continue.")
        st.stop()

    prep_col1, prep_col2, prep_col3 = st.columns(3)
    with prep_col1:
        missing_strategy = st.selectbox(
            "Missing-value handling",
            [
                "None",
                "Drop rows with NaN",
                "Fill numeric with mean + categorical with mode",
                "Fill numeric with median + categorical with mode",
            ],
            help=(
                "Choose how to handle missing values. For this dataset, median/mode is usually the safest option."
            ),
        )
    with prep_col2:
        scaling_strategy = st.selectbox(
            "Scaling for numeric features",
            ["None", "MinMax", "Standard (Z-score)"],
            help="Scaling helps many algorithms compare features on similar ranges.",
        )
    with prep_col3:
        encoding_strategy = st.selectbox(
            "Encoding for categorical features",
            ["None", "One-Hot", "Label Encoding"],
            help="One-Hot is safer for non-ordered categories (e.g., team names).",
        )

    setup_a, setup_b, setup_c = st.columns(3)
    setup_a.caption(f"Target: `{target_col}`")
    setup_b.caption(f"Features selected: `{len(selected_features)}`")
    setup_c.caption(f"Missing strategy: `{missing_strategy}`")

    model_df = df_raw[[target_col] + selected_features].copy()
    model_df = apply_missing_values(model_df, missing_strategy)
    eda_df = model_df.copy()

    X = model_df[selected_features].copy()
    y = model_df[target_col].copy()

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    if scaling_strategy != "None" and numeric_features:
        scaler = MinMaxScaler() if scaling_strategy == "MinMax" else StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    if encoding_strategy == "One-Hot" and categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    elif encoding_strategy == "Label Encoding" and categorical_features:
        for col in categorical_features:
            X[col] = X[col].astype("category").cat.codes

    processed_df = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    st.write("Processed data preview")
    st.dataframe(processed_df.head(), use_container_width=True)
    st.caption(f"Processed shape: {processed_df.shape[0]} rows x {processed_df.shape[1]} columns")

with tabs[1]:
    st.subheader("Exploratory Data Analysis (EDA)")
    with st.expander("What to do in this stage", expanded=False):
        st.markdown(
            """
            1. Check feature distributions and outliers.
            2. Inspect correlations to spot related stats.
            3. Use PCA to view player patterns in 2D.
            """
        )

    numeric_eda_features = eda_df[selected_features].select_dtypes(include="number").columns.tolist()

    if y.nunique(dropna=True) <= 20:
        st.write("Target/Class distribution")
        class_counts = y.astype(str).value_counts().reset_index()
        class_counts.columns = [target_col, "count"]
        fig_cls = px.bar(
            class_counts,
            x=target_col,
            y="count",
            title=f"Distribution of {target_col}",
        )
        fig_cls.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(style_fig(fig_cls, 390), use_container_width=True)
        plot_comment("Tall bars mean more samples in that class. Check if one class dominates.")

    st.write("Feature distributions (histograms)")
    default_hist = [c for c in ["pts_per_game", "ast_per_game", "trb_per_game", "mp_per_game"] if c in numeric_eda_features]
    if not default_hist:
        default_hist = numeric_eda_features[:4]

    hist_features = st.multiselect(
        "Select numeric features for histograms",
        options=numeric_eda_features,
        default=default_hist,
    )
    hist_bins = st.slider(
        "Histogram bins",
        min_value=10,
        max_value=80,
        value=30,
        step=5,
        help="Higher bins show finer detail, lower bins show broader distribution shape.",
    )
    if hist_features:
        for feat in hist_features:
            fig_hist = px.histogram(
                eda_df,
                x=feat,
                nbins=hist_bins,
                marginal="box",
                title=f"Histogram: {feat}",
            )
            st.plotly_chart(style_fig(fig_hist, 380), use_container_width=True)
            plot_comment("Look for skewness and extreme values. The box marginal shows spread and outliers.")
    else:
        st.info("Select at least one numeric feature for histogram plots.")

    st.write("Boxplots by target")
    box_default = [c for c in ["pts_per_game", "ast_per_game", "trb_per_game"] if c in numeric_eda_features]
    if not box_default:
        box_default = numeric_eda_features[:3]

    box_features = st.multiselect(
        "Select numeric features for boxplots",
        options=numeric_eda_features,
        default=box_default,
        key="box_features",
    )
    if box_features and y.nunique(dropna=True) <= 20:
        for feat in box_features:
            fig_box = px.box(
                eda_df,
                x=target_col,
                y=feat,
                color=target_col,
                title=f"{feat} by {target_col}",
            )
            fig_box.update_layout(xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(style_fig(fig_box, 390), use_container_width=True)
            plot_comment("Compare medians and spread across classes. Bigger separation means stronger predictive signal.")
    elif box_features:
        st.info("Boxplots by target are shown when target has <= 20 unique values.")

    st.write("Correlation heatmap")
    corr_input = eda_df[selected_features].select_dtypes(include="number")
    if corr_input.shape[1] >= 2:
        corr_matrix = corr_input.corr()
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Feature Correlations",
            aspect="auto",
        )
        st.plotly_chart(style_fig(fig_corr, 500), use_container_width=True)
        plot_comment("Values near +1/-1 indicate strong positive/negative linear relationships.")
    else:
        st.info("Need at least 2 numeric features for correlation heatmap.")

    st.write("2D dimensionality reduction (PCA)")
    pca_input = X.copy()
    if pca_input.select_dtypes(exclude="number").shape[1] > 0:
        pca_input = pd.get_dummies(pca_input, drop_first=False)

    if pca_input.shape[1] >= 2 and pca_input.shape[0] >= 3:
        # PCA cannot handle NaNs, so we impute any remaining missing values.
        pca_input = pca_input.replace([float("inf"), float("-inf")], pd.NA)
        pca_input = pca_input.dropna(axis=1, how="all")
        if pca_input.shape[1] >= 2:
            pca_input = pca_input.fillna(pca_input.median(numeric_only=True)).fillna(0)

            pca_scaled = StandardScaler().fit_transform(pca_input)
            pca = PCA(n_components=2, random_state=42)
            pca_points = pca.fit_transform(pca_scaled)

            pca_df = pd.DataFrame(pca_points, columns=["PC1", "PC2"], index=pca_input.index)
            pca_df["target"] = y.astype(str).values
            if "player" in df_raw.columns:
                pca_df["player"] = df_raw.loc[pca_df.index, "player"].astype(str).values
            if "season" in df_raw.columns:
                pca_df["season"] = df_raw.loc[pca_df.index, "season"].astype(str).values

            pca_plot_df = pca_df.copy()
            if len(pca_plot_df) > 1500:
                max_points = st.slider(
                    "PCA max points to display",
                    min_value=300,
                    max_value=len(pca_plot_df),
                    value=min(1500, len(pca_plot_df)),
                    step=100,
                    help="Reduce points to keep the plot responsive during interaction.",
                )
                pca_plot_df = pca_plot_df.sample(max_points, random_state=42)

            color_col = "target"
            symbol_col = None
            if "player" in pca_plot_df.columns:
                player_options = sorted(pca_plot_df["player"].dropna().unique().tolist())
                selected_player = st.selectbox(
                    "Highlight player in PCA",
                    options=["None"] + player_options[:500],
                    help="Optional: mark one player to inspect their location in the 2D projection.",
                )
                if selected_player != "None":
                    pca_plot_df["highlight"] = pca_plot_df["player"].apply(
                        lambda name: "Highlighted" if name == selected_player else "Other"
                    )
                    symbol_col = "highlight"

            fig_pca = px.scatter(
                pca_plot_df,
                x="PC1",
                y="PC2",
                color=color_col,
                symbol=symbol_col,
                opacity=0.75,
                title="PCA Projection (2D)",
                hover_data=[col for col in ["player", "season", "target"] if col in pca_plot_df.columns],
            )
            st.plotly_chart(style_fig(fig_pca, 520), use_container_width=True)
            plot_comment("Points close together have similar feature profiles after dimensionality reduction.")
            st.caption(
                f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.2f}, "
                f"PC2={pca.explained_variance_ratio_[1]:.2f}"
            )
        else:
            st.info("Not enough valid features for PCA after cleaning NaN/Inf values.")
    else:
        st.info("Need at least 2 usable features and 3 rows for PCA.")

with tabs[2]:
    st.subheader("ML Pipeline")
    with st.expander("What to do in this stage", expanded=False):
        st.markdown(
            """
            1. Train two supervised algorithms and compare metrics.
            2. Inspect confusion matrix (classification) or actual-vs-predicted (regression).
            3. Run clustering and compare `k` values with silhouette/elbow plots.
            """
        )
    st.markdown("#### Process 1: Supervised Learning")
    st.caption("Train and compare two algorithms on the selected target.")

    ml_df = pd.concat([X, y.rename("target")], axis=1).dropna()
    if ml_df.shape[0] < 10:
        st.error("Not enough valid rows after preprocessing for training.")
        st.stop()

    X_ml = ml_df.drop(columns=["target"])
    y_ml = ml_df["target"]

    if X_ml.select_dtypes(exclude="number").shape[1] > 0:
        X_ml = pd.get_dummies(X_ml, drop_first=False)

    pipeline_signature = (
        uploaded_file.name,
        tuple(active_seasons) if active_seasons is not None else (),
        target_col,
        tuple(selected_features),
        missing_strategy,
        scaling_strategy,
        encoding_strategy,
        int(ml_df.shape[0]),
        int(ml_df.shape[1]),
    )

    inferred_task = infer_task_type(y_ml)
    default_task = inferred_task
    if is_player_per_game_schema and goal_mode == "Predict player stats (regression)":
        default_task = "Regression"
    elif is_player_per_game_schema and goal_mode == "Predict player position (classification)":
        default_task = "Classification"
    task_type = st.radio(
        "Task type",
        ["Classification", "Regression"],
        index=0 if default_task == "Classification" else 1,
        help="Choose classification for categorical targets or regression for numeric targets.",
    )
    test_size = st.slider(
        "Test set size",
        0.1,
        0.4,
        0.2,
        0.05,
        help="Portion of data reserved for evaluation. 0.2 is a standard baseline.",
    )

    supervised_params = {}
    if task_type == "Classification":
        clf_col1, clf_col2, clf_col3 = st.columns(3)
        with clf_col1:
            c_value = st.slider("Logistic Regression: C", 0.01, 5.0, 1.0, 0.01, help="Lower C = stronger regularization.")
        with clf_col2:
            n_estimators_clf = st.slider("Random Forest: n_estimators", 50, 500, 200, 50, help="More trees can improve stability but increase runtime.")
        with clf_col3:
            max_depth_clf = st.slider("Random Forest: max_depth (0 means None)", 0, 30, 0, 1, help="Limits tree complexity to reduce overfitting.")
        supervised_params = {
            "c_value": float(c_value),
            "n_estimators_clf": int(n_estimators_clf),
            "max_depth_clf": int(max_depth_clf),
        }
    else:
        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            n_estimators_reg = st.slider("Random Forest: n_estimators", 50, 500, 200, 50, help="More trees can improve accuracy but increase runtime.")
        with reg_col2:
            max_depth_reg = st.slider("Random Forest: max_depth (0 means None)", 0, 30, 0, 1, help="Restrict depth to control overfitting.")
        supervised_params = {
            "n_estimators_reg": int(n_estimators_reg),
            "max_depth_reg": int(max_depth_reg),
        }

    supervised_config = (
        pipeline_signature,
        task_type,
        float(test_size),
        tuple(sorted(supervised_params.items())),
    )
    run_col, _ = st.columns([1, 1.2])
    with run_col:
        run_supervised = st.button(
            "Run Supervised Training",
            type="primary",
            use_container_width=True,
            key="run_supervised_btn",
            help="Click to train models with the current settings.",
        )

    if run_supervised:
        split_stratify = y_ml if task_type == "Classification" and y_ml.nunique(dropna=True) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_ml, y_ml, test_size=test_size, random_state=42, stratify=split_stratify
        )

        if task_type == "Classification":
            clf_a = LogisticRegression(max_iter=2000, C=supervised_params["c_value"])
            clf_b = RandomForestClassifier(
                n_estimators=supervised_params["n_estimators_clf"],
                max_depth=None if supervised_params["max_depth_clf"] == 0 else supervised_params["max_depth_clf"],
                random_state=42,
            )
            clf_a.fit(X_train, y_train)
            clf_b.fit(X_train, y_train)

            pred_a = clf_a.predict(X_test)
            pred_b = clf_b.predict(X_test)
            results = pd.DataFrame(
                {
                    "Algorithm": ["Logistic Regression", "Random Forest"],
                    "Accuracy": [accuracy_score(y_test, pred_a), accuracy_score(y_test, pred_b)],
                    "F1 (macro)": [
                        f1_score(y_test, pred_a, average="macro", zero_division=0),
                        f1_score(y_test, pred_b, average="macro", zero_division=0),
                    ],
                }
            )
            best_idx = results["Accuracy"].idxmax()
            best_name = results.loc[best_idx, "Algorithm"]
            best_pred = pred_a if best_name == "Logistic Regression" else pred_b

            labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
            cm = confusion_matrix(y_test.astype(str), pd.Series(best_pred).astype(str), labels=labels)

            pred_df = pd.DataFrame(
                {
                    "row_index": y_test.index,
                    "actual": y_test.astype(str).values,
                    "pred_logistic_regression": pd.Series(pred_a, index=y_test.index).astype(str).values,
                    "pred_random_forest": pd.Series(pred_b, index=y_test.index).astype(str).values,
                }
            )
            if "player" in df_raw.columns:
                pred_df["player"] = df_raw.loc[pred_df["row_index"], "player"].astype(str).values
            if "season" in df_raw.columns:
                pred_df["season"] = df_raw.loc[pred_df["row_index"], "season"].values

            st.session_state["supervised_result"] = {
                "task_type": "Classification",
                "results": results,
                "best_name": best_name,
                "labels": labels,
                "cm": cm,
                "predictions_df": pred_df.sort_values("row_index"),
                "best_accuracy": float(results.loc[best_idx, "Accuracy"]),
                "best_f1": float(results.loc[best_idx, "F1 (macro)"]),
            }
        else:
            reg_a = LinearRegression()
            reg_b = RandomForestRegressor(
                n_estimators=supervised_params["n_estimators_reg"],
                max_depth=None if supervised_params["max_depth_reg"] == 0 else supervised_params["max_depth_reg"],
                random_state=42,
            )
            reg_a.fit(X_train, y_train)
            reg_b.fit(X_train, y_train)

            pred_a = reg_a.predict(X_test)
            pred_b = reg_b.predict(X_test)
            rmse_a = mean_squared_error(y_test, pred_a) ** 0.5
            rmse_b = mean_squared_error(y_test, pred_b) ** 0.5

            results = pd.DataFrame(
                {
                    "Algorithm": ["Linear Regression", "Random Forest Regressor"],
                    "RMSE": [rmse_a, rmse_b],
                    "MAE": [mean_absolute_error(y_test, pred_a), mean_absolute_error(y_test, pred_b)],
                    "R2": [r2_score(y_test, pred_a), r2_score(y_test, pred_b)],
                }
            )
            best_idx = results["R2"].idxmax()
            best_name = results.loc[best_idx, "Algorithm"]
            best_pred = pred_a if best_name == "Linear Regression" else pred_b

            reg_plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": best_pred})
            pred_df = pd.DataFrame(
                {
                    "row_index": y_test.index,
                    "actual": y_test.values,
                    "pred_linear_regression": pd.Series(pred_a, index=y_test.index).values,
                    "pred_random_forest": pd.Series(pred_b, index=y_test.index).values,
                }
            )
            if "player" in df_raw.columns:
                pred_df["player"] = df_raw.loc[pred_df["row_index"], "player"].astype(str).values
            if "season" in df_raw.columns:
                pred_df["season"] = df_raw.loc[pred_df["row_index"], "season"].values

            st.session_state["supervised_result"] = {
                "task_type": "Regression",
                "results": results,
                "best_name": best_name,
                "reg_plot_df": reg_plot_df,
                "predictions_df": pred_df.sort_values("row_index"),
                "best_r2": float(results.loc[best_idx, "R2"]),
                "best_rmse": float(results.loc[best_idx, "RMSE"]),
            }

        st.session_state["supervised_config"] = supervised_config

    supervised_result = st.session_state.get("supervised_result")
    supervised_ready = supervised_result is not None and st.session_state.get("supervised_config") == supervised_config

    if not supervised_ready:
        if supervised_result is not None:
            st.warning("Supervised settings changed. Click `Run Supervised Training` to refresh results.")
        else:
            st.info("Click `Run Supervised Training` to compute and visualize supervised results.")
    else:
        if supervised_result["task_type"] == "Classification":
            results = supervised_result["results"]
            st.dataframe(results, use_container_width=True)

            metrics_plot = results.melt(
                id_vars="Algorithm",
                value_vars=["Accuracy", "F1 (macro)"],
                var_name="Metric",
                value_name="Score",
            )
            fig_cmp = px.bar(
                metrics_plot,
                x="Metric",
                y="Score",
                color="Algorithm",
                barmode="group",
                range_y=[0, 1],
                title="Classification Algorithm Comparison",
            )
            st.plotly_chart(style_fig(fig_cmp, 400), use_container_width=True)
            plot_comment("Higher Accuracy and Macro-F1 is better. Macro-F1 treats all classes more evenly.")

            st.write(f"Confusion Matrix ({supervised_result['best_name']})")
            fig_cm = px.imshow(
                supervised_result["cm"],
                x=supervised_result["labels"],
                y=supervised_result["labels"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix ({supervised_result['best_name']})",
                aspect="auto",
            )
            fig_cm.update_xaxes(title_text="Predicted")
            fig_cm.update_yaxes(title_text="Actual")
            st.plotly_chart(style_fig(fig_cm, 470), use_container_width=True)
            plot_comment("Diagonal cells are correct predictions. Off-diagonal cells show confusion between classes.")
        else:
            results = supervised_result["results"]
            st.dataframe(results, use_container_width=True)

            reg_metrics = results.melt(
                id_vars="Algorithm",
                value_vars=["RMSE", "MAE"],
                var_name="Metric",
                value_name="Value",
            )
            fig_reg_cmp = px.bar(
                reg_metrics,
                x="Metric",
                y="Value",
                color="Algorithm",
                barmode="group",
                title="Regression Algorithm Comparison",
            )
            st.plotly_chart(style_fig(fig_reg_cmp, 400), use_container_width=True)
            plot_comment("Lower RMSE/MAE is better. Compare bars to pick the stronger regressor.")

            reg_plot_df = supervised_result["reg_plot_df"]
            min_val = float(min(reg_plot_df["Actual"].min(), reg_plot_df["Predicted"].min()))
            max_val = float(max(reg_plot_df["Actual"].max(), reg_plot_df["Predicted"].max()))
            fig_reg = px.scatter(
                reg_plot_df,
                x="Actual",
                y="Predicted",
                opacity=0.7,
                title=f"Actual vs Predicted ({supervised_result['best_name']})",
            )
            fig_reg.add_shape(
                type="line",
                x0=min_val,
                y0=min_val,
                x1=max_val,
                y1=max_val,
                line=dict(color="red", dash="dash"),
            )
            st.plotly_chart(style_fig(fig_reg, 460), use_container_width=True)
            plot_comment("Points closer to the red diagonal line indicate more accurate predictions.")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download Supervised Metrics (CSV)",
                data=df_to_csv_bytes(supervised_result["results"]),
                file_name="supervised_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "Download Supervised Predictions (CSV)",
                data=df_to_csv_bytes(supervised_result["predictions_df"]),
                file_name="supervised_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### Process 2: Unsupervised Learning (KMeans)")
    st.caption("Explore player groups and evaluate cluster quality.")

    cluster_input = X_ml.copy()
    cluster_input = cluster_input.replace([float("inf"), float("-inf")], pd.NA)
    cluster_input = cluster_input.dropna(axis=1, how="all")
    cluster_input = cluster_input.fillna(cluster_input.median(numeric_only=True)).fillna(0)
    if cluster_input.shape[1] < 2:
        st.info("Need at least 2 features for clustering.")
        st.stop()

    cluster_scaled = StandardScaler().fit_transform(cluster_input)
    max_allowed_k = min(10, cluster_input.shape[0] - 1)
    if max_allowed_k < 2:
        st.info("Not enough samples for clustering.")
        st.stop()

    k_value = st.slider(
        "KMeans: number of clusters (k)",
        2,
        max_allowed_k,
        min(4, max_allowed_k),
        help="Try multiple k values and compare silhouette/elbow plots below.",
    )
    cluster_config = (pipeline_signature, int(k_value))
    run_cluster_col, _ = st.columns([1, 1.2])
    with run_cluster_col:
        run_clustering = st.button(
            "Run Clustering",
            type="secondary",
            use_container_width=True,
            key="run_clustering_btn",
            help="Click to run KMeans and refresh clustering visuals.",
        )

    if run_clustering:
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_scaled)
        silhouette_val = silhouette_score(cluster_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else float("nan")

        pca_cluster = PCA(n_components=2, random_state=42).fit_transform(cluster_scaled)
        cluster_df = pd.DataFrame(pca_cluster, columns=["PC1", "PC2"], index=cluster_input.index)
        cluster_df["cluster"] = cluster_labels.astype(str)

        assign_df = pd.DataFrame({"row_index": cluster_input.index, "cluster": cluster_labels})
        if "player" in df_raw.columns:
            assign_df["player"] = df_raw.loc[assign_df["row_index"], "player"].astype(str).values
        if "season" in df_raw.columns:
            assign_df["season"] = df_raw.loc[assign_df["row_index"], "season"].values
        assign_df["target"] = ml_df.loc[assign_df["row_index"], "target"].values

        score_df = pd.DataFrame()
        inertia_df = pd.DataFrame()
        if max_allowed_k >= 3:
            compare_end = min(8, max_allowed_k)
            k_grid = list(range(2, compare_end + 1))
            k_scores = []
            k_inertia = []
            for k in k_grid:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(cluster_scaled)
                score = silhouette_score(cluster_scaled, labels) if len(set(labels)) > 1 else float("nan")
                k_scores.append({"k": k, "silhouette": score})
                k_inertia.append({"k": k, "inertia": model.inertia_})
            score_df = pd.DataFrame(k_scores)
            inertia_df = pd.DataFrame(k_inertia)

        st.session_state["clustering_result"] = {
            "k_value": int(k_value),
            "inertia": float(kmeans.inertia_),
            "silhouette": float(silhouette_val) if pd.notna(silhouette_val) else float("nan"),
            "cluster_df": cluster_df,
            "assign_df": assign_df.sort_values("row_index"),
            "score_df": score_df,
            "inertia_df": inertia_df,
        }
        st.session_state["clustering_config"] = cluster_config

    clustering_result = st.session_state.get("clustering_result")
    clustering_ready = clustering_result is not None and st.session_state.get("clustering_config") == cluster_config

    if not clustering_ready:
        if clustering_result is not None:
            st.warning("Clustering settings changed. Click `Run Clustering` to refresh results.")
        else:
            st.info("Click `Run Clustering` to compute and visualize unsupervised results.")
    else:
        st.write(f"Inertia: {clustering_result['inertia']:.3f}")
        if pd.notna(clustering_result["silhouette"]):
            st.write(f"Silhouette Score: {clustering_result['silhouette']:.3f}")
        else:
            st.write("Silhouette Score: not available (single cluster found).")

        fig_cluster = px.scatter(
            clustering_result["cluster_df"],
            x="PC1",
            y="PC2",
            color="cluster",
            title="KMeans Clusters projected with PCA",
            opacity=0.75,
        )
        st.plotly_chart(style_fig(fig_cluster, 500), use_container_width=True)
        plot_comment("Each color is a cluster. Compact and clearly separated groups indicate better clustering.")

        if not clustering_result["score_df"].empty:
            st.write("KMeans configuration comparison")
            st.dataframe(clustering_result["score_df"], use_container_width=True)

            fig_k = px.line(
                clustering_result["score_df"],
                x="k",
                y="silhouette",
                markers=True,
                title="Silhouette Score vs k",
            )
            fig_k.update_xaxes(dtick=1)
            st.plotly_chart(style_fig(fig_k, 380), use_container_width=True)
            plot_comment("Higher silhouette is better; peak values suggest a better cluster count.")

            fig_elbow = px.line(
                clustering_result["inertia_df"],
                x="k",
                y="inertia",
                markers=True,
                title="Elbow Plot (Inertia vs k)",
            )
            fig_elbow.update_xaxes(dtick=1)
            st.plotly_chart(style_fig(fig_elbow, 380), use_container_width=True)
            plot_comment("Look for an elbow point where inertia reduction starts to flatten.")

        dlc1, dlc2 = st.columns(2)
        with dlc1:
            st.download_button(
                "Download Cluster Assignments (CSV)",
                data=df_to_csv_bytes(clustering_result["assign_df"]),
                file_name="cluster_assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dlc2:
            if clustering_result["score_df"].empty:
                st.caption("k-comparison table is available when at least 3 cluster values can be tested.")
            else:
                st.download_button(
                    "Download K Comparison (CSV)",
                    data=df_to_csv_bytes(clustering_result["score_df"]),
                    file_name="kmeans_k_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.divider()
    st.subheader("Pipeline Summary")
    summary_cols = st.columns(4)
    if supervised_ready:
        if supervised_result["task_type"] == "Classification":
            summary_cols[0].metric("Best Supervised Model", supervised_result["best_name"])
            summary_cols[1].metric("Best Accuracy", f"{supervised_result['best_accuracy']:.3f}")
            summary_cols[2].metric("Best Macro-F1", f"{supervised_result['best_f1']:.3f}")
        else:
            summary_cols[0].metric("Best Supervised Model", supervised_result["best_name"])
            summary_cols[1].metric("Best R2", f"{supervised_result['best_r2']:.3f}")
            summary_cols[2].metric("Best RMSE", f"{supervised_result['best_rmse']:.3f}")
    else:
        summary_cols[0].metric("Best Supervised Model", "Not computed")
        summary_cols[1].metric("Best Metric", "-")
        summary_cols[2].metric("Secondary Metric", "-")

    if clustering_ready:
        summary_cols[3].metric("KMeans (k / silhouette)", f"{clustering_result['k_value']} / {clustering_result['silhouette']:.3f}" if pd.notna(clustering_result["silhouette"]) else f"{clustering_result['k_value']} / n.a.")
    else:
        summary_cols[3].metric("KMeans (k / silhouette)", "Not computed")
