"""
visualization.py
Step 4 - Generate one meaningful chart automatically, based on the
dataset's column types. Uses Plotly for clean, interactive charts.
"""

import plotly.express as px
import pandas as pd
from analysis import get_numeric_columns, get_categorical_columns


def generate_chart(df: pd.DataFrame, chart_type: str = "Auto"):
    """
    Returns (plotly_figure, explanation_text).
    chart_type: "Auto", "Bar", "Pie", "Line", "Histogram", "Scatter"
    """
    numeric_cols = get_numeric_columns(df, exclude_ids=True)
    cat_cols = get_categorical_columns(df)

    # Prefer columns that look like the "main" measure (sales, revenue, price...)
    priority_keywords = ("sales", "revenue", "amount", "price", "total", "value", "quantity", "cost")
    if numeric_cols:
        preferred = [c for c in numeric_cols if any(k in c.lower() for k in priority_keywords)]
        if preferred:
            numeric_cols = preferred + [c for c in numeric_cols if c not in preferred]

    if chart_type == "Auto":
        if cat_cols and numeric_cols:
            chart_type = "Bar"
        elif numeric_cols:
            chart_type = "Histogram"
        elif cat_cols:
            chart_type = "Pie"
        else:
            return None, "No suitable columns found to visualize."

    colors = px.colors.qualitative.Set2

    if chart_type == "Bar" and cat_cols and numeric_cols:
        cat_col, num_col = cat_cols[0], numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(
            grouped, x=cat_col, y=num_col,
            title=f"Total {num_col} by {cat_col}",
            color=cat_col, color_discrete_sequence=colors,
            labels={cat_col: cat_col, num_col: f"Total {num_col}"},
        )
        top_row = grouped.iloc[0]
        share = top_row[num_col] / grouped[num_col].sum() * 100
        explanation = (
            f"'{top_row[cat_col]}' leads with a total {num_col} of {top_row[num_col]:,.2f}, "
            f"accounting for roughly {share:.1f}% of the top-10 categories shown."
        )

    elif chart_type == "Pie" and cat_cols:
        cat_col = cat_cols[0]
        counts = df[cat_col].value_counts().head(8).reset_index()
        counts.columns = [cat_col, "count"]
        fig = px.pie(
            counts, names=cat_col, values="count",
            title=f"Distribution of {cat_col}",
            color_discrete_sequence=colors,
        )
        top = counts.iloc[0]
        share = top["count"] / counts["count"].sum() * 100
        explanation = (
            f"'{top[cat_col]}' is the most common value in {cat_col}, making up about "
            f"{share:.1f}% of the records shown."
        )

    elif chart_type == "Histogram" and numeric_cols:
        num_col = numeric_cols[0]
        fig = px.histogram(
            df, x=num_col, title=f"Distribution of {num_col}",
            color_discrete_sequence=colors, nbins=20,
        )
        explanation = (
            f"{num_col} ranges from {df[num_col].min():,.2f} to {df[num_col].max():,.2f}, "
            f"with an average of {df[num_col].mean():,.2f}."
        )

    elif chart_type == "Line" and numeric_cols:
        num_col = numeric_cols[0]
        fig = px.line(
            df.reset_index(), x="index", y=num_col,
            title=f"{num_col} Over Records",
            color_discrete_sequence=colors,
        )
        explanation = f"{num_col} trends across the dataset, averaging {df[num_col].mean():,.2f}."

    elif chart_type == "Scatter" and len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        fig = px.scatter(
            df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
            color_discrete_sequence=colors,
        )
        corr = df[x_col].corr(df[y_col])
        explanation = f"{x_col} and {y_col} show a correlation coefficient of {corr:.2f}."

    else:
        return None, "Not enough suitable columns for the requested chart type."

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        font=dict(size=13),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig, explanation
