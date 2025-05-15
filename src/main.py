import os
import streamlit as st
import pandas as pd
from analyzer.text_analyzer import TextAnalyzer

def main():
    st.title("Topic Clustering Tool")
    st.write("Upload your Excel file and configure clustering parameters")
    
    analyzer = TextAnalyzer()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Read Excel file directly from the uploaded file
        df = pd.read_excel(uploaded_file)
        available_columns = df.columns.tolist()
        
        # Initialize session state for clustering results if not exists
        if 'clustering_results' not in st.session_state:
            st.session_state.clustering_results = None
        
        # Form for parameters
        with st.form("clustering_params"):
            # Column selection
            text_column = st.selectbox(
                "Select the column for topic clustering",
                options=available_columns
            )
            
            # Clustering parameters
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
            with col2:
                batch_size = st.number_input(
                    "Batch size",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100
                )
            
            # Submit button
            submitted = st.form_submit_button("Start Clustering")
            
            if submitted:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress, status=""):
                    progress_bar.progress(progress)
                    if status:
                        status_text.text(status)

                # Run clustering with progress updates
                st.session_state.clustering_results = analyzer.perform_topic_clustering(
                    df,  # Pass DataFrame directly instead of file path
                    text_column,
                    None,
                    threshold,
                    batch_size,
                    progress_callback=update_progress
                )
                
                if st.session_state.clustering_results is not None:
                    progress_bar.progress(1.0)
                    status_text.text("Topic clustering complete!")
                    st.success("Topic clustering complete!")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("Topic clustering failed. Please check your inputs and try again.")
        
        # Display download button outside the form if results exist
        if st.session_state.clustering_results is not None:
            # Create Excel file in memory
            import io
            excel_buffer = io.BytesIO()
            st.session_state.clustering_results.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            
            # Provide download button for results
            st.download_button(
                label="Download Results",
                data=excel_buffer,
                file_name="topic_clustering_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()