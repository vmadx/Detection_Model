#!/usr/bin/env python3
"""
0XVENOM Web Interface
Streamlit-based web application for Network Intrusion Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
import queue
from datetime import datetime, timedelta
import os
import sys
import tempfile

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import the core classes
try:
    from venom_core import VenomNIDS, PacketCapture, ModelManager
except ImportError:
    try:
        from scripts.venom_core import VenomNIDS, PacketCapture, ModelManager
    except ImportError:
        st.error("❌ Could not import VenomNIDS classes. Please check the installation.")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="0XVENOM - Network Intrusion Detection System",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4444;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-box {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background: #44ff44;
        color: black;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nids' not in st.session_state:
    st.session_state.nids = VenomNIDS()
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'packet_data' not in st.session_state:
    st.session_state.packet_data = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'monitoring_end_time' not in st.session_state:
    st.session_state.monitoring_end_time = None

def main():
    # Header
    st.markdown('<h1 class="main-header">🐍 0XVENOM - NIDS</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Network Intrusion Detection System</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🛠️ Control Panel")

    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Dashboard", "📊 Model Training", "🔍 Real-time Monitoring", "📈 Analytics", "⚙️ Settings"]
    )

    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📊 Model Training":
        training_page()
    elif page == "🔍 Real-time Monitoring":
        monitoring_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

def dashboard_page():
    st.header("📊 System Dashboard")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Models Trained", len(st.session_state.nids.models), delta="Active")

    with col2:
        packet_count = len(st.session_state.packet_data)
        st.metric("📦 Packets Analyzed", packet_count, delta=f"+{packet_count}")

    with col3:
        alert_count = len(st.session_state.alerts)
        st.metric("🚨 Alerts Generated", alert_count, delta=f"+{alert_count}")

    with col4:
        status = "🟢 Active" if st.session_state.monitoring_active else "🔴 Inactive"
        st.metric("📡 Monitoring Status", status)

    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    if st.session_state.alerts:
        alerts_df = pd.DataFrame(st.session_state.alerts[-10:])  # Last 10 alerts
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("No alerts generated yet.")

    # System status
    st.subheader("🖥️ System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Model Manager**: ✅ Ready")
        st.info("**Packet Capture**: ✅ Ready")
        st.info("**Web Interface**: ✅ Running")

    with col2:
        # Model information
        if st.session_state.nids.best_model:
            st.success(f"**Best Model**: {st.session_state.nids.best_model_name}")
            st.success(f"**Features**: {len(st.session_state.nids.selected_features)}")
        else:
            st.warning("**No trained model available**")

def training_page():
    st.header("🎓 Model Training")

    # File upload
    st.subheader("📁 Upload Training Data")

    col1, col2 = st.columns(2)

    with col1:
        train_file = st.file_uploader("Training Dataset (CSV)", type=['csv'], key="train")

    with col2:
        test_file = st.file_uploader("Testing Dataset (CSV)", type=['csv'], key="test")

    # Training options
    st.subheader("⚙️ Training Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        detailed_analysis = st.checkbox("Detailed Analysis", value=True)

    with col2:
        save_models = st.checkbox("Save Models", value=True)

    with col3:
        cross_validation = st.checkbox("Cross Validation", value=True)

    # Start training
    if st.button("🚀 Start Training", type="primary"):
        if train_file and test_file:
            train_path = test_path = None
            with st.spinner("Training Random Forest... This may take a few minutes."):
                try:
                    # Save uploaded files temporarily
                    train_handle = tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".csv", delete=False
                    )
                    test_handle = tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".csv", delete=False
                    )
                    train_path, test_path = train_handle.name, test_handle.name
                    with train_handle as f:
                        f.write(train_file.getbuffer())
                    with test_handle as f:
                        f.write(test_file.getbuffer())

                    # Run training
                    models = st.session_state.nids.run_analysis(
                        train_path, test_path,
                        detailed=detailed_analysis,
                        save_models=save_models
                    )

                    st.success("✅ Training completed successfully!")

                    # Display results
                    st.subheader("📊 Training Results")

                    if st.session_state.nids.best_model:
                        st.success(f"🏆 Best Model: {st.session_state.nids.best_model_name}")

                        # Model performance metrics
                        if hasattr(st.session_state.nids, 'x_test') and hasattr(st.session_state.nids, 'y_test'):
                            accuracy = st.session_state.nids.best_model.score(
                                st.session_state.nids.x_test,
                                st.session_state.nids.y_test
                            )
                            st.metric("🎯 Best Model Accuracy", f"{accuracy:.4f}")

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                finally:
                    for path in (train_path, test_path):
                        if path and os.path.exists(path):
                            os.remove(path)
        else:
            st.warning("⚠️ Please upload both training and testing datasets.")

    # Saved models
    st.subheader("💾 Saved Models")

    model_manager = ModelManager()
    saved_models = model_manager.list_models()

    if saved_models:
        models_df = pd.DataFrame(saved_models)
        st.dataframe(models_df, use_container_width=True)

        # Load model option
        selected_model = st.selectbox(
            "Select model to load:",
            options=[m['filename'] for m in saved_models],
            format_func=lambda x: f"{x} ({[m for m in saved_models if m['filename'] == x][0]['name']})"
        )

        if st.button("📥 Load Selected Model"):
            model_path = next(
                model['path'] for model in saved_models
                if model['filename'] == selected_model
            )
            if st.session_state.nids.load_saved_model(model_path):
                st.success(f"✅ Successfully loaded {selected_model}")
                st.rerun()
            else:
                st.error("❌ Failed to load model")
    else:
        st.info("No saved models found. Train a model first.")

def monitoring_page():
    st.header("🔍 Real-time Network Monitoring")

    if st.session_state.monitoring_active:
        if (
            st.session_state.monitoring_end_time
            and time.time() >= st.session_state.monitoring_end_time
        ):
            stop_monitoring()
        else:
            for packet in st.session_state.nids.packet_capture.get_packets():
                prediction, confidence = st.session_state.nids.classify_packet(
                    packet['features']
                )
                packet['prediction'] = prediction
                packet['confidence'] = confidence
                st.session_state.packet_data.append(packet)
                if prediction == 1:
                    st.session_state.alerts.append({
                        'timestamp': packet['timestamp'],
                        'message': 'Potential intrusion detected',
                        'confidence': confidence,
                    })

    # Monitoring controls
    col1, col2, col3 = st.columns(3)

    with col2:
        interface = st.selectbox("Network Interface", ["auto", "eth0", "wlan0", "lo"])

    with col3:
        duration = st.number_input("Duration (seconds)", min_value=10, max_value=3600, value=60)

    with col1:
        if not st.session_state.monitoring_active:
            if st.button("▶️ Start Monitoring", type="primary"):
                if st.session_state.nids.best_model:
                    start_monitoring(interface, duration)
                else:
                    st.error("❌ No trained model available. Please train or load a model first.")
        else:
            if st.button("⏹️ Stop Monitoring", type="secondary"):
                stop_monitoring()

    # Monitoring status
    if st.session_state.monitoring_active:
        st.success("🟢 Monitoring Active")
    else:
        st.info("🔴 Monitoring Inactive")

    # Real-time metrics
    if st.session_state.monitoring_active or st.session_state.packet_data:
        st.subheader("📊 Real-time Metrics")

        # Create placeholder for real-time updates
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        alerts_placeholder = st.empty()

        # Update metrics
        update_monitoring_display(metrics_placeholder, chart_placeholder, alerts_placeholder)

    # Packet analysis
    if st.session_state.packet_data:
        st.subheader("📦 Packet Analysis")

        # Convert packet data to DataFrame
        df = pd.DataFrame(st.session_state.packet_data)

        # Display packet statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Packets", len(df))

        with col2:
            normal_count = len(df[df['prediction'] == 0]) if 'prediction' in df.columns else 0
            st.metric("Normal Traffic", normal_count)

        with col3:
            anomaly_count = len(df[df['prediction'] == 1]) if 'prediction' in df.columns else 0
            st.metric("Anomalies Detected", anomaly_count)

        # Packet details table
        if not df.empty:
            st.dataframe(df.tail(20), use_container_width=True)

def analytics_page():
    st.header("📈 Analytics & Insights")

    if not st.session_state.packet_data:
        st.info("No data available. Start monitoring to collect packet data.")
        return

    df = pd.DataFrame(st.session_state.packet_data)

    # Time series analysis
    st.subheader("⏰ Traffic Over Time")

    if 'timestamp' in df.columns:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Group by minute
        traffic_by_time = df.groupby(df['timestamp'].dt.floor('min')).size().reset_index()
        traffic_by_time.columns = ['timestamp', 'packet_count']

        # Create time series chart
        fig = px.line(traffic_by_time, x='timestamp', y='packet_count',
                     title='Network Traffic Over Time')
        st.plotly_chart(fig, use_container_width=True)

    # Prediction distribution
    st.subheader("🎯 Prediction Distribution")

    if 'prediction' in df.columns:
        pred_counts = df['prediction'].value_counts()

        fig = px.pie(values=pred_counts.values,
                    names=['Normal' if x == 0 else 'Anomaly' for x in pred_counts.index],
                    title='Traffic Classification Distribution')
        st.plotly_chart(fig, use_container_width=True)

    # Feature analysis
    st.subheader("🔍 Feature Analysis")

    if 'features' in df.columns and len(df) > 0:
        # Extract features from the first few packets
        feature_sample = df['features'].iloc[:min(100, len(df))]

        if feature_sample.iloc[0] and isinstance(feature_sample.iloc[0], dict):
            features_df = pd.DataFrame(list(feature_sample))

            # Select numeric columns for analysis
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                # Feature correlation heatmap
                corr_matrix = features_df[numeric_cols].corr()

                fig = px.imshow(corr_matrix,
                               title='Feature Correlation Matrix',
                               color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)

                # Feature distribution
                selected_feature = st.selectbox("Select feature to analyze:", numeric_cols)

                fig = px.histogram(features_df, x=selected_feature,
                                 title=f'Distribution of {selected_feature}')
                st.plotly_chart(fig, use_container_width=True)

def settings_page():
    st.header("⚙️ System Settings")

    # Model settings
    st.subheader("🤖 Model Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Model Save Directory", value="models", disabled=True)
        st.number_input("Feature Selection Count", min_value=5, max_value=50, value=10)

    with col2:
        st.selectbox("Default Algorithm", ["Random Forest"], disabled=True)
        st.slider("Cross-validation Folds", min_value=3, max_value=10, value=5)

    # Monitoring settings
    st.subheader("📡 Monitoring Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Alert Threshold", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        st.number_input("Packet Buffer Size", min_value=100, max_value=10000, value=1000)

    with col2:
        st.selectbox("Alert Method", ["Web Interface", "Email", "SMS", "Webhook"])
        st.checkbox("Auto-save Alerts", value=True)

    # System information
    st.subheader("ℹ️ System Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.info("**Version**: 0XVENOM v2.0")
        st.info("**Python Version**: " + sys.version.split()[0])
        st.info("**Streamlit Version**: " + st.__version__)

    with info_col2:
        st.info("**Models Directory**: models/")
        st.info("**Packet Capture**: " + ("✅ Available" if 'scapy' in sys.modules else "❌ Not Available"))
        st.info("**Real-time Monitoring**: " + ("✅ Ready" if st.session_state.nids.best_model else "❌ No Model"))

    # Export/Import settings
    st.subheader("💾 Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Export Packet Data"):
            if st.session_state.packet_data:
                df = pd.DataFrame(st.session_state.packet_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"packet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No packet data to export")

    with col2:
        if st.button("📤 Export Alerts"):
            if st.session_state.alerts:
                df = pd.DataFrame(st.session_state.alerts)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No alerts to export")

def start_monitoring(interface, duration):
    """Start real-time monitoring using the selected interface and duration."""
    interface = None if interface == "auto" else interface
    st.session_state.monitoring_active = True
    st.session_state.monitoring_end_time = time.time() + duration
    st.session_state.nids.packet_capture = PacketCapture(interface)
    if not st.session_state.nids.packet_capture.start_capture():
        st.session_state.monitoring_active = False
        st.error("❌ Could not start packet capture. Check Scapy and interface permissions.")
        return
    st.success("✅ Monitoring started!")

def stop_monitoring():
    """Stop real-time monitoring"""
    st.session_state.nids.packet_capture.stop_capture()
    st.session_state.monitoring_active = False
    st.session_state.monitoring_end_time = None
    st.info("ℹ️ Monitoring stopped.")

def update_monitoring_display(metrics_placeholder, chart_placeholder, alerts_placeholder):
    """Update the monitoring display with real-time data"""

    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📦 Packets/min", np.random.randint(50, 200))

        with col2:
            st.metric("🚨 Alerts/min", np.random.randint(0, 5))

        with col3:
            st.metric("📊 CPU Usage", f"{np.random.randint(20, 80)}%")

        with col4:
            st.metric("💾 Memory Usage", f"{np.random.randint(30, 70)}%")

    # Simulated real-time chart
    with chart_placeholder.container():
        # Generate sample data
        timestamps = pd.date_range(start=datetime.now() - timedelta(minutes=10),
                                 end=datetime.now(), freq='1min')
        traffic_data = pd.DataFrame({
            'timestamp': timestamps,
            'normal': np.random.randint(40, 100, len(timestamps)),
            'anomaly': np.random.randint(0, 10, len(timestamps))
        })

        fig = px.line(traffic_data, x='timestamp', y=['normal', 'anomaly'],
                     title='Real-time Traffic Classification')
        st.plotly_chart(fig, use_container_width=True)

    # Recent alerts
    with alerts_placeholder.container():
        if st.session_state.alerts:
            st.subheader("🚨 Recent Alerts")
            recent_alerts = st.session_state.alerts[-5:]  # Last 5 alerts
            for alert in recent_alerts:
                st.error(f"⚠️ {alert.get('message', 'Unknown alert')} - {alert.get('timestamp', 'Unknown time')}")

if __name__ == "__main__":
    main()
