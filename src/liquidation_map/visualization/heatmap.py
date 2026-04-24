"""Liquidation Heatmap Visualization with Plotly."""

from pathlib import Path

import plotly.graph_objects as go
import polars as pl


class LiquidationHeatmap:
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
    ):
        self.width = width
        self.height = height
    
    def create_bar_chart(
        self,
        df: pl.DataFrame,
        current_price: float,
        title: str = "Liquidation Map",
        symbol: str = "BTC",
    ) -> go.Figure:
        """
        Coinglass-style bidirectional bar chart.
        Longs (green) extend left, Shorts (red) extend right.
        """
        if df.is_empty():
            return self._empty_chart(title)
        
        df_sorted = df.sort("price_bucket")
        prices = df_sorted["price_bucket"].to_list()
        longs = df_sorted["long_volume"].to_list()
        shorts = df_sorted["short_volume"].to_list()
        
        max_volume = max(max(longs) if longs else 0, max(shorts) if shorts else 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=prices,
            x=[-v for v in longs],
            orientation="h",
            name="Long Liquidations",
            marker_color="rgba(0, 200, 83, 0.8)",
            hovertemplate="Price: $%{y:,.0f}<br>Long Volume: $%{customdata:,.0f}<extra></extra>",
            customdata=longs,
        ))
        
        fig.add_trace(go.Bar(
            y=prices,
            x=shorts,
            orientation="h",
            name="Short Liquidations",
            marker_color="rgba(255, 82, 82, 0.8)",
            hovertemplate="Price: $%{y:,.0f}<br>Short Volume: $%{x:,.0f}<extra></extra>",
        ))
        
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="yellow",
            line_width=2,
            annotation_text=f"Current: ${current_price:,.0f}",
            annotation_position="right",
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_title="Liquidation Volume (USD)",
            yaxis_title=f"{symbol} Price (USD)",
            barmode="overlay",
            width=self.width,
            height=self.height,
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis=dict(
                tickformat=",.0f",
                range=[-max_volume * 1.1, max_volume * 1.1] if max_volume > 0 else [-1, 1],
            ),
            yaxis=dict(tickformat=",.0f"),
        )
        
        return fig
    
    def create_heatmap(
        self,
        df_timeseries: pl.DataFrame,
        current_price: float,
        title: str = "Liquidation Heatmap Over Time",
    ) -> go.Figure:
        """
        2D heatmap: X=time, Y=price, color=volume.
        Requires DataFrame with: timestamp, price_bucket, volume columns.
        """
        if df_timeseries.is_empty():
            return self._empty_chart(title)
        
        pivot = df_timeseries.pivot(
            on="timestamp",
            index="price_bucket", 
            values="total_volume",
        ).fill_null(0)
        
        price_buckets = pivot["price_bucket"].to_list()
        timestamps = [c for c in pivot.columns if c != "price_bucket"]
        z_values = pivot.select(timestamps).to_numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=timestamps,
            y=price_buckets,
            colorscale="Hot",
            colorbar=dict(title="Volume (USD)"),
            hovertemplate="Time: %{x}<br>Price: $%{y:,.0f}<br>Volume: $%{z:,.0f}<extra></extra>",
        ))
        
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="cyan",
            line_width=2,
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            width=self.width,
            height=self.height,
            template="plotly_dark",
        )
        
        return fig
    
    def create_cumulative_chart(
        self,
        df: pl.DataFrame,
        current_price: float,
        title: str = "Cumulative Liquidations",
    ) -> go.Figure:
        """Area chart showing cumulative liquidation volume from current price."""
        if df.is_empty():
            return self._empty_chart(title)
        
        df_below = df.filter(pl.col("price_bucket") < current_price).sort("price_bucket", descending=True)
        df_above = df.filter(pl.col("price_bucket") > current_price).sort("price_bucket")
        
        fig = go.Figure()
        
        if not df_below.is_empty():
            prices_below = df_below["price_bucket"].to_list()
            cum_longs = df_below["long_volume"].cum_sum().to_list()
            
            fig.add_trace(go.Scatter(
                x=[(current_price - p) / current_price * 100 for p in prices_below],
                y=cum_longs,
                fill="tozeroy",
                name="Long Liquidations",
                line_color="rgba(0, 200, 83, 0.8)",
                fillcolor="rgba(0, 200, 83, 0.3)",
            ))
        
        if not df_above.is_empty():
            prices_above = df_above["price_bucket"].to_list()
            cum_shorts = df_above["short_volume"].cum_sum().to_list()
            
            fig.add_trace(go.Scatter(
                x=[(p - current_price) / current_price * 100 for p in prices_above],
                y=cum_shorts,
                fill="tozeroy",
                name="Short Liquidations",
                line_color="rgba(255, 82, 82, 0.8)",
                fillcolor="rgba(255, 82, 82, 0.3)",
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="% Move from Current Price",
            yaxis_title="Cumulative Volume (USD)",
            width=self.width,
            height=self.height,
            template="plotly_dark",
            showlegend=True,
        )
        
        return fig
    
    def _empty_chart(self, title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text="No liquidation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20),
        )
        fig.update_layout(
            title=dict(text=title, x=0.5),
            width=self.width,
            height=self.height,
            template="plotly_dark",
        )
        return fig
    
    def save(self, fig: go.Figure, output_path: str | Path, format: str = "html") -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "html":
            fig.write_html(str(output_path))
        elif format == "png":
            fig.write_image(str(output_path))
        elif format == "json":
            fig.write_json(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")
