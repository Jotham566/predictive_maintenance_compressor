import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from typing import List, Tuple, Union

from scipy import signal
from scipy.fft import fft, fftfreq

# Add imports at the top of the file
from functools import partial
from frozendict import frozendict
import streamlit as st

# Set page config
st.set_page_config(page_title="Sound Analysis Dashboard", layout="wide")

# --- Theme Settings ---
primary_color = "#2c3e50"  # Dark Slate Gray
secondary_color = "#3498db" # Peter River Blue
background_color = "#f5f5f5" # Light Gray
text_color = "#2e2e2e"      # Dark Gray

st.markdown(
    f"""
    <style>
        body {{
            color: {text_color};
            background-color: {background_color};
        }}
        .stApp {{
            background-color: {background_color};
            padding-top: 20px;
            padding-left: 30px;
            padding-right: 30px;
        }}
        .st-emotion-cache-sidebar {{
            background-color: #ecf0f1;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {primary_color};
        }}
        h1 {{
            margin-bottom: 30px;
        }}
        .stButton>button {{
            color: white;
            background-color: {secondary_color};
            border-color: {secondary_color};
        }}
        .stSlider>div>div>div>div {{
            background-color: {secondary_color};
        }}
        .stSelectbox>div>div>div {{
            color: {text_color};
        }}
        .stMultiSelect>div>div>div {{
            color: {text_color};
        }}
        .streamlit-expanderHeader {{
            font-weight: bold;
            color: {primary_color};
        }}
        .css-1egvi7u {{
            background-color: #bdc3c7;
            color: {text_color};
            font-weight: bold;
        }}
        .css-1rs6os {{
            background-color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("Fridge/Compressor Sound Recording Failure Analysis Dashboard")
st.markdown("""
This dashboard visualizes clustering analysis of sound recordings across different days.
Use the controls in the sidebar to customize the visualization and compare day ranges.

### Instructions:
- **Select Days**: You can select days as a range (e.g., 1-10), a single day (e.g., 5), or multiple days separated by commas (e.g., 1, 5, 10).
- **Clustering Analysis**: Compare multiple day ranges by entering each range on a new line in the sidebar.
- **Deterioration Analysis**: Analyze daily gravity (mean) centers and distances.
- **Experiment Data**: Easily switch between different experiment data folders using the dropdown in the sidebar.
""")
st.markdown("---") # Add separator line below description

# Helper function to make data structures hashable for caching
def make_hashable(obj):
    """Convert a container hierarchy into one that can be hashed for caching."""
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return frozendict({key: make_hashable(value) for key, value in obj.items()})
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)  # Fallback for other types

# Add this utility function for caching with unhashable inputs
def cache_with_unhashable(func):
    """Decorator to make a function cacheable even with unhashable inputs."""
    @st.cache_data
    def wrapped(*args, **kwargs):
        hashable_args = tuple(make_hashable(arg) for arg in args)
        hashable_kwargs = {key: make_hashable(value) for key, value in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

def parse_event_line(line):
    """
    Parse a single event line with proper handling of rgba() color values.
    Returns a dictionary with the parsed event data or raises ValueError with a specific error message.
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
        
    try:
        # First, find any rgba() or rgb() color definitions
        import re
        color_match = re.search(r'(rgba?\([^)]+\))', line)
        if not color_match:
            raise ValueError("No valid color definition found")
            
        color_str = color_match.group(1)
        
        # Replace the color temporarily with a placeholder
        temp_line = line.replace(color_str, 'COLOR_PLACEHOLDER')
        
        # Split the rest of the line by commas and clean up the parts
        parts = [p.strip().strip('"').strip("'") for p in temp_line.split(',')]
        
        # Validate required fields
        if len(parts) < 5:
            raise ValueError("Missing required fields. Format: day,text,y,color,angle[,vline]")
            
        # Parse and validate each field
        try:
            day = int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid day value: {parts[0]}")
            
        text = parts[1]
        
        try:
            y = float(parts[2])
        except ValueError:
            raise ValueError(f"Invalid y value: {parts[2]}")
            
        try:
            angle = int(parts[4])
        except ValueError:
            raise ValueError(f"Invalid angle value: {parts[4]}")
            
        # Handle the optional vline parameter
        vline = True
        if len(parts) > 5:
            vline_str = parts[5].lower().strip()
            if vline_str not in ['true', 'false']:
                raise ValueError(f"Invalid vline value: {parts[5]}. Use 'true' or 'false'")
            vline = vline_str != 'false'
            
        return {
            "day": day,
            "text": text,
            "y": y,
            "color": color_str,
            "angle": angle,
            "vline": vline
        }
    except ValueError as e:
        raise ValueError(str(e))
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")

def parse_day_input(day_input_str: str) -> Union[List[int], None]:
    """
    Parse a day input string which can be a range (e.g., '1-10'), a single day ('5'),
    or comma-separated days ('1,5,10'). Returns a list of integers representing the selected days.
    """
    days = []
    try:
        if '-' in day_input_str:
            start_str, end_str = day_input_str.split('-')
            start_day = int(start_str)
            end_day = int(end_str)
            if start_day > end_day:
                st.error(f"Invalid day range: start day must be less than or equal to end day.")
                return None
            days = list(range(start_day, end_day + 1))
        elif ',' in day_input_str:
            day_str_list = day_input_str.split(',')
            days = [int(day.strip()) for day in day_str_list]
        else:
            days = [int(day_input_str)]
        return days
    except ValueError:
        st.error(f"Invalid day input format: {day_input_str}. Please use format 'start-end', 'day', or 'day1, day2, ...'")
        return None

def parse_multiple_day_inputs(day_input_text: str) -> List[List[int]]:
    """
    Parse a text input containing multiple day ranges, one per line.
    Returns a list of lists of days for each valid range.
    """
    day_ranges_list = []
    input_lines = day_input_text.strip().split('\n')
    for line in input_lines:
        if line.strip(): # Ignore empty lines
            days = parse_day_input(line.strip())
            if days: # Only add valid day ranges
                day_ranges_list.append(days)
    return day_ranges_list


@st.cache_data
def filter_data_by_days(df: pd.DataFrame, selected_days: List[int]) -> pd.DataFrame:
    """Filter dataframe for days within the specified list of days."""
    return df[df['Day'].isin(selected_days)].copy()

@st.cache_data
def load_data(results_folder, filename):
    """Load data from CSV file."""
    return pd.read_csv(os.path.join(results_folder, filename))

@st.cache_data
def compute_frequency_spectrum(signal_data, sampling_rate=8000, nperseg=4096):
    """Compute the frequency spectrum of the signal using Welch's method."""
    freqs, psd = signal.welch(signal_data,
                             fs=sampling_rate,
                             nperseg=nperseg,
                             scaling='spectrum')

    # Convert to dB
    psd_db = 10 * np.log10(psd)

    return freqs, psd_db

def plot_frequency_spectrum(freqs, spectrum, title="Frequency Spectrum"):
    """Create a plotly figure of the frequency spectrum."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=spectrum,
        mode='lines',
        name='Spectrum',
        line=dict(color=secondary_color, width=2) # Use secondary color
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family="Arial", color=primary_color) # Use primary color
        ),
        xaxis=dict(
            title="Frequency (Hz)",
            type='log',
            gridcolor='rgba(189,195,199,0.4)',
            showline=True,
            linecolor='rgba(189,195,199,0.6)'
        ),
        yaxis=dict(
            title="Power Spectral Density (dB/Hz)",
            gridcolor='rgba(189,195,199,0.4)',
            showline=True,
            linecolor='rgba(189,195,199,0.6)'
        ),
        plot_bgcolor='rgba(245,245,245,0.95)',
        paper_bgcolor='white',
        height=600,
        showlegend=False
    )

    return fig

def plot_spectrogram(signal_data, sampling_rate=8000, title="Spectrogram"):
    """Create a spectrogram plot."""
    f, t, Sxx = signal.spectrogram(signal_data, fs=sampling_rate)

    fig = go.Figure(data=go.Heatmap(
        x=t,
        y=f,
        z=10 * np.log10(Sxx),
        colorscale='Viridis'
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family="Arial", color=primary_color) # Use primary color
        ),
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Frequency (Hz)", type='log'),
        height=400
    )

    return fig

@st.cache_data
def enhance_clustering_viz(df, x_col="x_dinsight", y_col="y_dinsight",
                         file_col="Day", day_col="Day",
                         bandwidth=0.1, point_size_scale=20,
                         title_suffix=""):
    """Create enhanced clustering visualization."""
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure numeric data
    df.loc[:, file_col] = df[file_col].astype(float)
    df.loc[:, day_col] = df[day_col].astype(int)

    # Kernel Density Estimation
    X = df[[x_col, y_col]].values
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(X)
    density = np.exp(kde.score_samples(X))

    # Normalize density for marker sizing
    size = 10 + (density - density.min()) / (density.max() - density.min()) * point_size_scale

    # Prepare custom data columns
    df.loc[:, 'Day_Label'] = 'Day ' + df[day_col].astype(str)

    # Create base scatter plot
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        hover_data={
            file_col: True,
            day_col: ':.0f',
            x_col: ':.3f',
            y_col: ':.3f',
            'File_No': True
        },
        hover_name=day_col,
        custom_data=['Day_Label', 'File_No'],
        color=file_col,
        color_continuous_scale="turbo", # Reverted to 'turbo' colorscale
        size=size,
        size_max=30
    )

    # Update hover template with enhanced styling
    fig.update_traces(
        hovertemplate="<b style='font-size: 14px; color: {primary_color}'>%{customdata[0]}</b><br>" + # Use primary color
        "<span style='font-size: 12px; color: #7f8c8d'>Recording Number: </span>" +
        "<b style='color: {primary_color}'>%{customdata[1]}</b><br>" + # Use primary color
        f"<span style='font-size: 12px; color: #7f8c8d'>{x_col}: </span>" +
        "<b style='color: {primary_color}'>%{x:.3f}</b><br>" + # Use primary color
        f"<span style='font-size: 12px; color: #7f8c8d'>{y_col}: </span>" +
        "<b style='color: {primary_color}'>%{y:.3f}</b><br>" + # Use primary color
        "<extra></extra>",
        marker=dict(
            line=dict(width=1, color='rgba(255,255,255,0.8)'),
            opacity=0.85
        )
    )

    # Add density contours
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_range = np.linspace(df[y_col].min(), df[y_col].max(), 100)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    density_grid = np.exp(kde.score_samples(positions)).reshape(100, 100)

    contour = go.Contour(
        z=density_grid,
        x=x_range,
        y=y_range,
        colorscale=[[0, 'rgba(255,255,255,0)'], [1, 'rgba(189,195,199,0.4)']],
        opacity=0.4,
        showscale=False,
        hoverinfo='skip',
        contours=dict(
            showlines=True,
            coloring='fill',
            showlabels=False,
            start=density_grid.min(),
            end=density_grid.max(),
            size=(density_grid.max() - density_grid.min()) / 10
        )
    )
    fig.add_trace(contour)

    # Update layout with enhanced styling
    fig.update_layout(
        height=800,
        width=900,
        plot_bgcolor='rgba(245,245,245,0.95)',
        paper_bgcolor='white',
        title=dict(
            text=f'Sound Recording Clustering Analysis {title_suffix}',
            x=0.5,
            font=dict(size=20, family="Arial", color=primary_color) # Use primary color
        ),
        xaxis=dict(
            title=dict(text="x_dinsight", font=dict(size=14, family="Arial", color=primary_color)), # Use primary color
            gridcolor='rgba(189,195,199,0.4)',
            gridwidth=1,
            showline=True,
            linecolor='rgba(189,195,199,0.6)',
            linewidth=1,
            mirror=True,
            zeroline=True,
            zerolinecolor='#e0e0e0', # Lighter zeroline - Corrected to single entry
            zerolinewidth=1,
        ),
        yaxis=dict(
            title=dict(text="y_dinsight", font=dict(size=14, family="Arial", color=primary_color)), # Use primary color
            gridcolor='rgba(189,195,199,0.4)',
            gridwidth=1,
            showline=True,
            linecolor='rgba(189,195,199,0.6)',
            linewidth=1,
            mirror=True,
            zeroline=True,
            zerolinecolor='#e0e0e0', # Lighter zeroline - Corrected to single entry
            zerolinewidth=1,
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text="Sound Recording Days",
                font=dict(size=12, family="Arial", color=primary_color) # Use primary color
            ),
            tickfont=dict(size=10, family="Arial", color=primary_color), # Use primary color
            len=0.8,
            thickness=20,
            outlinewidth=1,
            outlinecolor='rgba(189,195,199,0.6)'
        ),
        legend_title_text="Day (24 files/Day)",
        legend=dict(
            font=dict(size=10, family="Arial", color=primary_color), # Use primary color
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(189,195,199,0.6)',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor='rgba(189,195,199,0.6)'
        ),
        margin=dict(l=80, r=80, t=100, b=80)
    )

    return fig

def display_statistics(df: pd.DataFrame, bandwidth: float, title: str = ""):
    """Display statistics for the dataset."""
    st.subheader(f"Statistics {title}") # Make statistics a subheader
    col1, col2 = st.columns(2) # Keep columns

    with col1:
        st.markdown("**Daily Statistics:**") # Bold title
        daily_stats = df.groupby('Day').agg({
            'x_dinsight': ['mean', 'std'],
            'y_dinsight': ['mean', 'std']
        }).round(3)
        st.dataframe(daily_stats, use_container_width=True) # Added use_container_width
        st.markdown("") # Add a bit of spacing below dataframe if needed

    with col2:
        st.markdown("**Cluster Density Information:**") # Bold title
        X = df[['x_dinsight', 'y_dinsight']].values
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X)
        density = np.exp(kde.score_samples(X))
        density_stats = pd.DataFrame({
            'Statistic': ['Min', 'Max', 'Mean', 'Std'],
            'Value': [
                density.min(),
                density.max(),
                density.mean(),
                density.std()
            ]
        }).round(4)
        st.dataframe(density_stats, use_container_width=True) # Added use_container_width
        st.markdown("") # Add a bit of spacing below dataframe if needed

def format_day_range_label(selected_days: List[int]) -> str:
    """Formats a list of selected days into a concise label for titles."""
    if not selected_days:
        return ""
    if len(selected_days) == 1:
        return f"Day {selected_days[0]}"

    selected_days.sort()
    is_consecutive = True
    for i in range(1, len(selected_days)):
        if selected_days[i] != selected_days[i - 1] + 1:
            is_consecutive = False
            break

    if is_consecutive:
        return f"Days {selected_days[0]}-{selected_days[-1]}"
    else:
        return f"Days {', '.join(map(str, selected_days))}"

@st.cache_data
def calculate_daily_centers_and_distances_mean(df, G0_cluster_days=None, x_col="x_dinsight", y_col="y_dinsight", day_col="Day", save_high_res=False,
                                             show_G0_cluster_shade=True, poor_cooling_regions=None, show_poor_cooling_shade=True,
                                             events=None, show_event_annotations=True):
    """
    Calculate Means and distances with enhanced control over plot elements.

    Args:
        df: DataFrame with the data.
        G0_cluster_days: Days for the G0 (benchmark) Cluster.
        x_col, y_col: Column names for coordinates.
        day_col: Column name for the day.
        save_high_res: Save high-resolution image.
        show_G0_cluster_shade: Enable/disable G0 (benchmark) Cluster shading.
        poor_cooling_regions: List of tuples (start_day, end_day) for poor cooling periods.
        show_poor_cooling_shade: Enable/disable poor cooling period shading.
        events: List of dictionaries for events (day, text, y, color, angle).
        show_event_annotations: Enable/disable event annotations.
    """

    if G0_cluster_days is None:
        G0_cluster_days = range(int(df[day_col].min()), int(df[day_col].max()) + 1)

    # Convert range to list for caching (ranges are not hashable)
    if isinstance(G0_cluster_days, range):
        G0_cluster_days = list(G0_cluster_days)
        
    # Convert poor_cooling_regions to a tuple of tuples for caching if it exists
    if poor_cooling_regions:
        poor_cooling_regions = tuple(tuple(region) for region in poor_cooling_regions)
        
    # Convert events to tuple of frozendict for caching if they exist
    if events:
        events = tuple(frozendict(event) for event in events)

    # Calculate daily averages (Gi) using mean
    daily_centers = df.groupby(day_col)[[x_col, y_col]].mean()

    # Calculate G0 using only the G0 Cluster Days
    G0_cluster_data = df[df[day_col].isin(G0_cluster_days)]
    G0 = G0_cluster_data[[x_col, y_col]].mean()

    # Calculate distances between G0 and Gi
    G0_distances = daily_centers.apply(lambda row: np.sqrt(
        (row[x_col] - G0[x_col])**2 +
        (row[y_col] - G0[y_col])**2
    ), axis=1)

    # Calculate G0 mean distance
    G0_mean_distance = G0_distances.mean()

    # Calculate distances between consecutive days (Gi and Gi+1)
    consecutive_distances = []
    for i in range(len(daily_centers)-1):
        # Use .iloc instead of positional indexing
        curr_day = daily_centers.iloc[i]
        next_day = daily_centers.iloc[i+1]
        distance = np.sqrt(
            (curr_day[x_col] - next_day[x_col])**2 +
            (curr_day[y_col] - next_day[y_col])**2
        )
        consecutive_distances.append({
            'Day': i+1,
            'Distance': distance
        })

    consecutive_distances_df = pd.DataFrame(consecutive_distances)
    consecutive_mean_distance = consecutive_distances_df['Distance'].mean()

    # Create the distance charts with more spacing
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Distance between G0 and Gi', 'Distance between Gi and Gi+1'),
        vertical_spacing=0.22
    )

    # Plot G0 to Gi distances with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=G0_distances.index,
            y=G0_distances.values,
            mode='lines+markers',
            name='G0 to Gi Distance',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # Add mean line with enhanced visibility
    fig.add_hline(
        y=G0_mean_distance,
        line_dash="dash",
        line_color="rgba(128, 128, 128, 0.8)",
        line_width=2,
        row=1, col=1
    )

    # Enhanced mean annotation
    fig.add_annotation(
        x=45,
        y=G0_mean_distance,
        text=f"G0 Mean Distance: {G0_mean_distance:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(128, 128, 128, 0.8)",
        font=dict(size=12, color="rgba(128, 128, 1)"),
        align="left",
        row=1, col=1,
        ax=50,
        ay=-30
    )

    # Plot Gi to Gi+1 distances with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=consecutive_distances_df['Day'],
            y=consecutive_distances_df['Distance'],
            mode='lines+markers',
            name='Gi to Gi+1 Distance',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )

    # Enhanced consecutive mean line
    fig.add_hline(
        y=consecutive_mean_distance,
        line_dash="dash",
        line_color="rgba(128, 128, 128, 0.8)",
        line_width=2,
        row=2, col=1
    )

    # Enhanced consecutive mean annotation
    fig.add_annotation(
        x=45,
        y=consecutive_mean_distance,
        text=f"Consecutive Days Mean: {consecutive_mean_distance:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(128, 128, 128, 0.8)",
        font=dict(size=12, color="rgba(128, 128, 1)"),
        align="left",
        row=2, col=1,
        ax=50,
        ay=-30
    )

    # Add G0 (benchmark) Cluster annotation
    fig.add_annotation(
        x=0.82,
        y=0.79,
        xref="paper",
        yref="paper",
        text=f"G0 (benchmark) Cluster: Days {min(G0_cluster_days)}-{max(G0_cluster_days)}",
        showarrow=False,
        font=dict(size=12, color="rgba(0, 0, 0, 0.8)"),
        xanchor="left",
        yanchor="top"
    )

    # Add shading for G0 (benchmark) Cluster - Conditional Shading
    if show_G0_cluster_shade:
        fig.add_vrect(
            x0=min(G0_cluster_days), x1=max(G0_cluster_days),
            fillcolor="rgba(200, 255, 200, 0.4)", layer="below", line_width=0, row=1, col=1
        )
        fig.add_vrect(
            x0=min(G0_cluster_days), x1=max(G0_cluster_days),
            fillcolor="rgba(200, 255, 200, 0.4)", layer="below", line_width=0, row=2, col=1
        )

    # Enhanced Poor Cooling Period shading - User-defined and Conditional Shading
    if show_poor_cooling_shade and poor_cooling_regions:
        for start_day, end_day in poor_cooling_regions:
            fig.add_vrect(
                x0=start_day, x1=end_day,
                fillcolor="rgba(255, 200, 200, 0.4)", layer="below", line_width=0, row=1, col=1
            )
            fig.add_vrect(
                x0=start_day, x1=end_day,
                fillcolor="rgba(255, 200, 200, 0.4)", layer="below", line_width=0, row=2, col=1
            )

    # Enhanced event annotations - User-defined and Conditional Annotations
    if show_event_annotations and events:
        for event in events:
            fig.add_annotation(
                x=event["day"], y=event["y"], text=event["text"], showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=event["color"],
                font=dict(size=12, color=event["color"]), align="left", row=1, col=1,
                ax=40 * np.cos(np.radians(event["angle"])), ay=40 * np.sin(np.radians(event["angle"]))
            )
            if event.get("vline", True): #Option to disable vline per event
                fig.add_vline(
                    x=event["day"], line_dash="dot", line_color=event["color"], line_width=2, opacity=0.7, row=1, col=1
                )
                fig.add_vline(
                    x=event["day"], line_dash="dot", line_color=event["color"], line_width=2, opacity=0.7, row=2, col=1
                )

    # Enhanced layout
    fig.update_layout(
        height=900,
        width=1000,
        title_text=f"Deterioration (Mean distance) Analysis:  (G0 (benchmark) Cluster: Days {min(G0_cluster_days)}-{max(G0_cluster_days)})",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    # Enhanced axes
    for row in [1, 2]:
        fig.update_xaxes(
            title_text="Day",
            title_font=dict(size=14),
            gridcolor='rgba(128,128,128,0.2)',
            gridwidth=1,
            row=row, col=1,
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.2)',
            zerolinewidth=2
        )
        fig.update_yaxes(
            title_text="Distance",
            title_font=dict(size=14),
            gridcolor='rgba(128,128,128,0.2)',
            gridwidth=1,
            row=row, col=1,
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.2)',
            zerolinewidth=2
        )

    # Save high-resolution image if requested
    if save_high_res:
        fig.write_image("gravity_center_ave.png", scale=-5,
                       width=2000, height=1800)

    return {
        'daily_centers': daily_centers,
        'G0': G0,
        'G0_distances': G0_distances,
        'G0_mean_distance': G0_mean_distance,
        'consecutive_distances': consecutive_distances_df,
        'consecutive_mean_distance': consecutive_mean_distance,
        'figure': fig
    }


def main():
    # Initialize variables that might be needed across different modes
    viz_mode = None
    G0_cluster_days = None
    show_center_shade = True
    poor_cooling_regions = []
    show_poor_cooling = False
    events = []
    show_event_annotations = False
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Data Parameters")

        # 1. Construct base path to Store relative to the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        store_dir = os.path.join(script_dir, '')

        # Check if Store exists
        if not os.path.exists(store_dir):
            st.error(f"Error: 'Store' directory not found at: {store_dir}")
            return

        # Get available Line folders
        line_folders = [f for f in os.listdir(store_dir) if os.path.isdir(os.path.join(store_dir, f)) and f.startswith('Line_')]
        if not line_folders:
            st.error("No Line folders found in Store directory")
            return

        # Line selector with default option
        line_options = ["Select Line"] + sorted(line_folders)
        selected_line = st.selectbox("Select Line", options=line_options)

        if selected_line == "Select Line":
            st.info("Please select a Line to proceed")
            return

        # Construct path to outputFiles for selected line
        output_files_dir = os.path.join(store_dir, selected_line, 'outputFiles')

        # Initialize results_folder
        results_folder = None
        originalDataDf = None
        
        # Check if outputFiles_dir exists before proceeding
        if not os.path.exists(output_files_dir):
            st.error(f"Error: 'outputFiles' directory not found at: {output_files_dir}")
            return
            
        experiment_folders = [f for f in os.listdir(output_files_dir) if os.path.isdir(os.path.join(output_files_dir, f))]
        if not experiment_folders:
            st.warning(f"No experiment folders found in '{output_files_dir}'. Please ensure experiment folders are present inside 'outputFiles'.")
            return
            
        exp_res = st.selectbox("Experiment Folder", options=experiment_folders)
        if not exp_res:
            st.info("Please select an experiment folder to proceed")
            return

        # Set up results folder and load data
        results_folder = os.path.join(output_files_dir, exp_res)
        readFileName = f"xy-dinsight-{exp_res}.csv"
        
        try:
            originalDataDf = load_data(results_folder, readFileName)
            min_day = int(originalDataDf['Day'].min())
            max_day = int(originalDataDf['Day'].max())

            # Visualization mode selection
            st.subheader("Visualization Mode")
            viz_mode = st.radio(
                "Select Visualization Mode",
                ["Clustering Analysis", "Deterioration Analysis"]
            )

            # Common visualization parameters
            st.subheader("Visualization Parameters")
            bandwidth = st.slider("KDE Bandwidth", 0.01, 1.0, 0.1, 0.01)
            point_size = st.slider("Point Size Scale", 5, 50, 20, 5)

            if viz_mode == "Deterioration Analysis":
                st.subheader("G0 (benchmark) Parameters")
                center_days_input = st.text_input(
                    "G0 Cluster Days (e.g., 1-10 or 1,2,3)", # For full days from start to end use f"{min_day}-{max_day}"
                    value=f"{min_day}-5"
                )
                G0_cluster_days = parse_day_input(center_days_input)
                if G0_cluster_days is None:
                    G0_cluster_days = range(min_day, max_day + 1)

                show_center_shade = st.checkbox("Show G0 (benchmark) Cluster Shading", value=True)

                st.subheader("Poor Cooling Period Shading")
                show_poor_cooling = st.checkbox("Show Poor Cooling Shading", value=False)

                poor_cooling_regions_input = st.text_area(
                    "Poor Cooling Regions (start-end day per line, e.g., 2-5)",
                    f"1-{max_day}",
                    height=80
                )
                parsed_poor_cooling_regions_input = [parse_day_input(line) for line in poor_cooling_regions_input.strip().split('\n') if line.strip()]
                poor_cooling_regions = []
                for days_list in parsed_poor_cooling_regions_input:
                    if days_list and len(days_list) >= 2:
                        poor_cooling_regions.append((min(days_list), max(days_list)))

                st.subheader("Event Annotations")
                show_events = st.checkbox("Show Event Annotations", value=False)
                if show_events:
                    events_input_str = st.text_area(
                        "Event Annotations (day,text,y,color,angle,vline[optional] per line)",
                        """9,"Poor Cooling Period",0.05,"rgba(255, 100, 100, 1)",-30,false
                        7,"Gas Replenishment(Oct/27th)",0.12,"rgba(100, 100, 255, 1)",30
                        14,"Repair & Gas Replenishment(Nov/4th)",0.10,"rgba(100, 200, 100, 1)",-45""",
                        height=150
                    )

                    events = []
                    for line in events_input_str.strip().split('\n'):
                        if not line.strip():
                            continue
                        try:
                            event = parse_event_line(line)
                            events.append(event)
                        except ValueError as e:
                            st.warning(f"Error in event line: '{line}'\nDetails: {str(e)}")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    # Main area content
    results_folder = os.path.join(output_files_dir, exp_res) if exp_res else None

    try:
        if results_folder and viz_mode:
            # Load clustering data
            readFileName = f"xy-dinsight-{exp_res}.csv"
            originalDataDf = load_data(results_folder, readFileName)

            # Dataset Information Expander
            with st.expander("Dataset Information", expanded=False):
                st.write("Data Shape:", originalDataDf.shape)
                st.write("Columns:", originalDataDf.columns.tolist())
                st.write("Sample Data:")
                st.dataframe(originalDataDf.head())

            if viz_mode == "Deterioration Analysis":
                st.header("Deterioration Analysis")
                if G0_cluster_days:
                    results = calculate_daily_centers_and_distances_mean(
                        originalDataDf,
                        G0_cluster_days=G0_cluster_days,
                        show_G0_cluster_shade=show_center_shade,
                        poor_cooling_regions=poor_cooling_regions,
                        show_poor_cooling_shade=show_poor_cooling,
                        events=events,
                        show_event_annotations=show_event_annotations
                    )
                    st.plotly_chart(results['figure'], use_container_width=True)

                    st.subheader("Deterioration Analysis Statistics")
                    st.markdown("### Mean Distance Statistics")
                    st.markdown(f"**G0 Cluster Days:** {format_day_range_label(list(G0_cluster_days))}")
                    st.metric(label="Overall (Benchmark Cluster Mean) (G0) - X", value=f"{results['G0']['x_dinsight']:.4f}")
                    st.metric(label="Overall (Benchmark Cluster Mean) (G0) - Y", value=f"{results['G0']['y_dinsight']:.4f}")
                    st.metric(label="Mean distance from G0 to daily centers (Gi)", value=f"{results['G0_mean_distance']:.4f}")
                    st.metric(label="Mean distance between consecutive days (Gi and Gi+1)", value=f"{results['consecutive_mean_distance']:.4f}")

                else:
                    st.error("Invalid G0 Cluster Days input.")

            elif viz_mode == "Clustering Analysis":
                st.header("Clustering Analysis")
                day_input_text = st.text_area(
                    "Enter Day Ranges (one per line, e.g., 1-15, 16-31, 45)",
                    f"1-{max_day}",
                    height=150
                )
                day_ranges_list = parse_multiple_day_inputs(day_input_text)

                if day_ranges_list:
                    num_ranges = len(day_ranges_list)
                    if num_ranges > 0:
                        cols = st.columns(num_ranges)

                    for i, selected_days in enumerate(day_ranges_list):
                        if selected_days:
                            filtered_df = filter_data_by_days(originalDataDf, selected_days)
                            range_label = format_day_range_label(selected_days)

                            with cols[i] if num_ranges > 0 else st:
                                st.markdown(f"### Range: {range_label}")
                                fig = enhance_clustering_viz(
                                    filtered_df,
                                    bandwidth=bandwidth,
                                    point_size_scale=point_size,
                                    title_suffix=f"({range_label})"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                display_statistics(filtered_df, bandwidth, title=f"for {range_label}")
                                st.markdown("---")

        elif not exp_res:
            st.warning("Please select an experiment folder from the sidebar to proceed.")

    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        if exp_res:
            st.info(f"Please ensure the correct file structure exists within the '{exp_res}' experiment folder.")
        else:
            st.info("Please select an experiment folder first.")

if __name__ == "__main__":
    main()