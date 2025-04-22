import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
from agent import Agent
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class SegmentationAgent(Agent):
    """
    Agent that uses K-means clustering and DeepSeek LLM to segment customers based on RFM data
    """
    name = "Segmentation Agent"
    color = Agent.CYAN
    MODEL = "deepseek-r1:1.5b"
    
    def __init__(self, output_dir="output"):
        """
        Set up this instance by connecting to DeepSeek and initializing segmentation state
        """
        self.log("Initializing Segmentation Agent")
        
        # Store output directory
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up the DeepSeek client via Ollama
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        # Segmentation state
        self.rfm_data = None
        self.segments = None
        self.segment_descriptions = None
        self.segment_strategies = None
        self.kmeans_model = None
        self.scaler = None
        self.min_clusters = 3  # Minimum number of clusters to ensure
        
        self.log(f"Segmentation Agent is ready with {self.MODEL}")
    
    def load_data(self, data_path: str) -> bool:
        """
        Load the RFM data for segmentation
        """
        self.log(f"Loading data from {data_path}")
        
        try:
            # Determine file type and load accordingly
            if data_path.endswith('.parquet'):
                self.rfm_data = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                self.rfm_data = pd.read_csv(data_path)
            else:
                self.log(f"Unsupported file format: {data_path}")
                return False
            
            # Ensure required columns exist
            required_columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
            for col in required_columns:
                if col not in self.rfm_data.columns:
                    self.log(f"Missing required column: {col}")
                    return False
            
            self.log(f"Data loaded successfully with {len(self.rfm_data)} customers")
            return True
        
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            return False
    
    def _get_rfm_summary(self) -> Dict[str, Any]:
        """Analyze RFM data and create a summary for the LLM"""
        summary = {}
        
        # Calculate basic statistics
        for metric in ['Recency', 'Frequency', 'Monetary']:
            summary[metric] = {
                "min": float(self.rfm_data[metric].min()),
                "max": float(self.rfm_data[metric].max()),
                "mean": float(self.rfm_data[metric].mean()),
                "median": float(self.rfm_data[metric].median()),
                "p25": float(self.rfm_data[metric].quantile(0.25)),
                "p75": float(self.rfm_data[metric].quantile(0.75))
            }
        
        # Calculate correlations
        corr_matrix = self.rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        summary["correlations"] = {
            "Recency_Frequency": float(corr_matrix.loc['Recency', 'Frequency']),
            "Recency_Monetary": float(corr_matrix.loc['Recency', 'Monetary']),
            "Frequency_Monetary": float(corr_matrix.loc['Frequency', 'Monetary'])
        }
        
        return summary
    
    def messages_for_cluster_interpretation(self, clusters: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create the message list for generating cluster interpretations with improved prompting
        """
        system_message = """You are an expert in customer segmentation and marketing analytics.
        Your task is to interpret K-means clusters from RFM (Recency, Frequency, Monetary) data 
        and provide meaningful segment names and descriptions. Follow the standard RFM segmentation
        framework used in marketing analytics."""
        
        user_message = f"""
        I have performed K-means clustering on customer RFM data and need your help interpreting the clusters.
        
        Here are the clusters with their average RFM values and other statistics:
        {json.dumps(clusters, indent=2)}
        
        Remember that:
        - Recency: Days since last purchase (LOWER is better)
        - Frequency: Number of purchases (HIGHER is better)
        - Monetary: Average spending per customer (HIGHER is better)
        
        Use these standard segment names when appropriate:
        - Champions: Recent purchases, frequent buyers, high spending
        - Loyal Customers: Buy regularly but not as recently as Champions
        - New Customers: Recent first purchase, low frequency/spend
        - At-Risk Customers: Above-average recency, good past purchase frequency
        - Big Spenders: High monetary values but lower frequency
        - Lost Customers: High recency (haven't purchased in a long time)
        
        For each cluster, please:
        1. Provide a descriptive segment name from the standard list above when applicable, or create a custom name if needed
        2. Write a detailed description of the customer behavior and characteristics
        3. Suggest specific marketing strategies tailored to this customer segment
        
        Format your response as a JSON object with the following structure:
        {{
            "renamed_clusters": {{
                "0": "Descriptive name for cluster 0",
                "1": "Descriptive name for cluster 1",
                ...
            }},
            "descriptions": {{
                "0": "Detailed description for cluster 0...",
                "1": "Detailed description for cluster 1...",
                ...
            }},
            "strategies": {{
                "0": "Marketing strategies for cluster 0...",
                "1": "Marketing strategies for cluster 1...",
                ...
            }}
        }}
        
        Be specific and insightful in your analysis. Use the RFM values to inform your interpretation.
        Return only the JSON object.
        """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "{"}
        ]
    
    def messages_for_report(self, segments: Dict[str, Any], descriptions: Dict[str, str], strategies: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Create the message list for generating a report
        """
        system_message = """You are a marketing analytics expert who creates comprehensive business reports.
        Create a detailed customer segmentation report in Markdown format."""
        
        user_message = f"""
        Create a comprehensive customer segmentation report based on the provided segmentation data. The report should include:

        1. Executive Summary
        2. Introduction and Methodology (mention that K-means clustering was used on RFM data)
        3. Overview of Customer Segments
        4. Detailed Analysis of Each Segment
        5. Marketing Recommendations
        6. Conclusion

        Here's the segmentation data:
        {json.dumps(segments)}

        Additional insights about each segment:
        Descriptions:
        {json.dumps(descriptions)}

        Marketing Strategies:
        {json.dumps(strategies)}

        Format the report in Markdown with proper headings, bullet points, and any tables/charts you think would be useful (described in Markdown).
        Make the report professional, insightful, and actionable for business stakeholders.
        """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def extract_json(self, content: str) -> Any:
        """Extract JSON from LLM response"""
        try:
            # Try to parse as-is first
            return json.loads(content)
        except:
            # Try to extract JSON if embedded in other text
            try:
                # For objects
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                # For arrays
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                return None
            except:
                return None
    
    def _find_optimal_k(self, X_scaled, max_k=10) -> int:
        """Find optimal number of clusters using silhouette score and elbow method"""
        self.log("Finding optimal number of clusters")
        
        if len(X_scaled) < self.min_clusters:
            return self.min_clusters
        
        max_k = min(max_k, len(X_scaled) - 1)
        silhouette_scores = []
        inertia_values = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=15, init='k-means++')
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertia_values.append(kmeans.inertia_)
            self.log(f"K={k}, Silhouette Score={silhouette_avg:.4f}, Inertia={kmeans.inertia_:.2f}")
        
        # Find the best k using both silhouette score and elbow method
        # Calculate inertia differences (for elbow method)
        inertia_diffs = np.diff(inertia_values)
        inertia_diffs = np.append(inertia_diffs, inertia_diffs[-1])  # pad last element
        
        # Normalize both metrics between 0 and 1
        norm_silhouette = np.array(silhouette_scores) / max(silhouette_scores)
        norm_inertia_diffs = inertia_diffs / max(inertia_diffs)
        
        # Combined score: high silhouette and significant inertia drop
        combined_score = norm_silhouette + norm_inertia_diffs
        
        optimal_k = k_range[np.argmax(combined_score)]
        self.log(f"Optimal number of clusters: {optimal_k}")
        
        # Force a minimum of desired clusters for better segmentation
        optimal_k = max(optimal_k, self.min_clusters)
        self.log(f"Final number of clusters (after enforcing minimum): {optimal_k}")
        return optimal_k
    
    def perform_segmentation(self, num_segments: int = 0) -> Dict[str, Any]:
        """
        Perform customer segmentation using K-means clustering
        """
        self.log(f"Starting K-means segmentation")
        
        if self.rfm_data is None:
            self.log("No data loaded. Please load data first.")
            return {}
        
        # Filter outliers first, similar to notebook approach
        filtered_data = self._filter_outliers(self.rfm_data)
        
        # Get RFM features for clustering
        X = filtered_data[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Scale features using RobustScaler (same as in notebook)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal number of clusters if not specified
        if num_segments <= 0:
            num_segments = self._find_optimal_k(X_scaled, max_k=8)
        
        # Perform K-means clustering with improved parameters
        self.log(f"Performing K-means clustering with {num_segments} clusters")
        self.kmeans_model = KMeans(
            n_clusters=num_segments,
            random_state=42,
            n_init=20,  # Increase number of initializations for better stability
            init='k-means++',
            max_iter=1000,  # Increased from 500 for better convergence
            tol=1e-5  # Tighter tolerance for better convergence
        )
        
        # Fit model on filtered data
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        filtered_data['segment_id'] = cluster_labels
        
        # Now propagate cluster assignments to the original dataset
        # Create a mapping from customer ID to cluster
        customer_to_cluster = dict(zip(filtered_data['Customer ID'], filtered_data['segment_id']))
        
        # Apply to the entire original dataset
        self.rfm_data['segment_id'] = self.rfm_data['Customer ID'].map(customer_to_cluster)
        
        # Fill any unassigned customers (those that were outliers)
        if self.rfm_data['segment_id'].isna().any():
            # For unassigned customers, predict their cluster
            missing_mask = self.rfm_data['segment_id'].isna()
            missing_customers = self.rfm_data[missing_mask]
            
            if len(missing_customers) > 0:
                missing_X = missing_customers[['Recency', 'Frequency', 'Monetary']].values
                missing_X_scaled = self.scaler.transform(missing_X)
                missing_labels = self.kmeans_model.predict(missing_X_scaled)
                self.rfm_data.loc[missing_mask, 'segment_id'] = missing_labels
        
        # Ensure segment_id is integer type
        self.rfm_data['segment_id'] = self.rfm_data['segment_id'].astype(int)
        
        # Debug clustering results
        unique, counts = np.unique(self.rfm_data['segment_id'], return_counts=True)
        cluster_distribution = dict(zip(unique, counts))
        self.log(f"Cluster distribution: {cluster_distribution}")
        
        # Create segment information
        segments = {}
        for i in range(num_segments):
            segment_mask = self.rfm_data['segment_id'] == i
            segment_data = self.rfm_data[segment_mask]
            
            # Add temporary generic names
            self.rfm_data.loc[segment_mask, 'segment_name'] = f"Cluster {i}"
            
            segments[str(i)] = {
                "segment_id": i,
                "segment_name": f"Cluster {i}",
                "count": int(segment_mask.sum()),
                "percentage": float(segment_mask.sum() / len(self.rfm_data) * 100),
                "center_recency": float(self.kmeans_model.cluster_centers_[i, 0]),
                "center_frequency": float(self.kmeans_model.cluster_centers_[i, 1]),
                "center_monetary": float(self.kmeans_model.cluster_centers_[i, 2]),
                "avg_recency": float(segment_data['Recency'].mean()),
                "avg_frequency": float(segment_data['Frequency'].mean()),
                "avg_monetary": float(segment_data['Monetary'].mean()),
                "total_monetary": float(segment_data['Monetary'].sum()),
                "std_recency": float(segment_data['Recency'].std()),
                "std_frequency": float(segment_data['Frequency'].std()),
                "std_monetary": float(segment_data['Monetary'].std()),
                "min_recency": float(segment_data['Recency'].min()),
                "max_recency": float(segment_data['Recency'].max()),
                "min_frequency": float(segment_data['Frequency'].min()),
                "max_frequency": float(segment_data['Frequency'].max()),
                "min_monetary": float(segment_data['Monetary'].min()),
                "max_monetary": float(segment_data['Monetary'].max())
            }
        
        self.segments = segments
        
        # Use LLM to interpret the clusters with improved prompting
        try:
            self.log(f"Calling {self.MODEL} to interpret clusters")
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for_cluster_interpretation(self.segments),
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            cluster_insights = self.extract_json(content)
            
            if (not cluster_insights or 
                    "renamed_clusters" not in cluster_insights or
                    "descriptions" not in cluster_insights or
                    "strategies" not in cluster_insights):
                self.log("Failed to parse cluster insights from LLM response")
                insights = self._get_fallback_insights(self.segments)
                self.segment_descriptions = insights["descriptions"]
                self.segment_strategies = insights["strategies"]
                
                # Use fallback names based on RFM patterns instead of generic names
                self._apply_rfm_based_names(self.segments)
            else:
                # Update segment names
                for segment_id, new_name in cluster_insights["renamed_clusters"].items():
                    if segment_id in self.segments:
                        self.segments[segment_id]["segment_name"] = new_name
                        # Update in the dataframe too
                        self.rfm_data.loc[self.rfm_data['segment_id'] == int(segment_id), 'segment_name'] = new_name
                
                self.segment_descriptions = cluster_insights["descriptions"]
                self.segment_strategies = cluster_insights["strategies"]
        except Exception as e:
            self.log(f"Error using LLM for cluster interpretation: {str(e)}")
            insights = self._get_fallback_insights(self.segments)
            self.segment_descriptions = insights["descriptions"]
            self.segment_strategies = insights["strategies"]
            self._apply_rfm_based_names(self.segments)
        
        # Compile results
        results = {
            "num_customers": len(self.rfm_data),
            "num_segments": len(self.segments),
            "segments": self.segments,
            "segment_descriptions": self.segment_descriptions,
            "segment_strategies": self.segment_strategies,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results
        results_file = os.path.join(self.output_dir, "kmeans_segmentation_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.log(f"Segmentation complete - created {len(self.segments)} segments with K-means")
        return results
    
    def _apply_rfm_based_names(self, segments):
        """Apply RFM-based names to clusters based on their characteristics"""
        # Sort segments by their RFM metrics for consistent naming
        sorted_segments = []
        for seg_id, segment in segments.items():
            # Calculate RFM scores (1-5 scale)
            r_score = 5 - min(int(segment['avg_recency'] / 60) + 1, 5)  # Lower recency is better
            f_score = min(int(segment['avg_frequency'] / 5) + 1, 5)  # Higher frequency is better
            m_score = min(int(segment['avg_monetary'] / 200) + 1, 5)  # Higher monetary is better
            
            rfm_score = r_score * 100 + f_score * 10 + m_score  # Combine for sorting
            sorted_segments.append((seg_id, segment, rfm_score, r_score, f_score, m_score))
        
        # Sort by RFM score descending
        sorted_segments.sort(key=lambda x: x[2], reverse=True)
        
        # Name segments based on their RFM patterns and relative position
        for i, (seg_id, segment, _, r_score, f_score, m_score) in enumerate(sorted_segments):
            if r_score >= 4 and f_score >= 4 and m_score >= 4:
                name = "Champions"
            elif r_score >= 4 and f_score >= 3:
                name = "Loyal Customers"
            elif r_score >= 4 and f_score <= 2:
                name = "New Customers"
            elif r_score <= 2 and f_score >= 3:
                name = "At-Risk Customers"
            elif r_score <= 2 and f_score <= 2 and m_score <= 2:
                name = "Lost Customers"
            elif m_score >= 4 and f_score <= 2:
                name = "Big Spenders"
            elif r_score <= 2 and m_score >= 4:
                name = "At-Risk High Value"
            else:
                name = "Regular Customers"
            
            # Update segment name
            segment["segment_name"] = name
            # Update in dataframe
            self.rfm_data.loc[self.rfm_data['segment_id'] == int(seg_id), 'segment_name'] = name
    
    def _filter_outliers(self, dataframe):
        """Filter outliers using IQR method, similar to notebook approach"""
        self.log("Filtering outliers using IQR method")
        df = dataframe.copy()
        
        for column in ['Recency', 'Frequency', 'Monetary']:
            Q1 = df[column].quantile(0.01)  # Use 1st percentile instead of 0.25 for less aggressive filtering
            Q3 = df[column].quantile(0.99)  # Use 99th percentile instead of 0.75
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter outliers
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        self.log(f"After outlier removal: {len(df)} customers (removed {len(dataframe) - len(df)} outliers)")
        return df
    
    def _get_fallback_insights(self, segments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback insights if LLM fails"""
        descriptions = {}
        strategies = {}
        
        for segment_id, segment in segments.items():
            # Create basic descriptions
            descriptions[segment_id] = (
                f"Cluster {segment_id}: This segment contains {segment['count']} customers ({segment['percentage']:.1f}% of total). "
                f"On average, they made their last purchase {segment['avg_recency']:.1f} days ago, "
                f"have made {segment['avg_frequency']:.1f} purchases, and "
                f"spent ${segment['avg_monetary']:.2f} per customer."
            )
            
            # Create basic strategies based on RFM values
            recency = segment['avg_recency']
            frequency = segment['avg_frequency']
            monetary = segment['avg_monetary']
            
            if recency < segments["0"]["avg_recency"] and frequency > segments["0"]["avg_frequency"]:
                strategies[segment_id] = (
                    "These are likely your best customers. Reward them with loyalty programs, exclusive offers, "
                    "and early access to new products. Focus on building relationships and turning them into brand advocates."
                )
            elif recency > segments["0"]["avg_recency"] and frequency > segments["0"]["avg_frequency"]:
                strategies[segment_id] = (
                    "These customers used to shop frequently but haven't returned recently. Implement a win-back campaign "
                    "with personalized offers based on their past purchases."
                )
            elif recency < segments["0"]["avg_recency"] and frequency < segments["0"]["avg_frequency"]:
                strategies[segment_id] = (
                    "These are likely new customers. Focus on onboarding and education about your products and services. "
                    "Cross-sell related products to increase their engagement."
                )
            else:
                strategies[segment_id] = (
                    "Develop personalized communication and offers based on their purchasing patterns. "
                    "Analyze their behavior to find opportunities to increase engagement and spending."
                )
        
        return {
            "descriptions": descriptions,
            "strategies": strategies
        }
    
    def generate_report(self, output_file: str = "customer_segmentation_report.md") -> str:
        """
        Generate a comprehensive customer segmentation report
        """
        self.log(f"Generating customer segmentation report")
        
        if self.segments is None:
            self.log("No segmentation results. Please perform segmentation first.")
            return ""
        
        # Call DeepSeek to generate the report
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for_report(
                    self.segments, 
                    self.segment_descriptions,
                    self.segment_strategies
                ),
                max_tokens=4096
            )
            
            report_content = response.choices[0].message.content.strip()
        except Exception as e:
            self.log(f"Error generating report: {str(e)}")
            report_content = self._generate_fallback_report()
        
        # Save the report
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, "w") as f:
            f.write(report_content)
        
        self.log(f"Report saved to {output_path}")
        return output_path
    
    def _generate_fallback_report(self) -> str:
        """Generate a simple fallback report if LLM fails"""
        report = []
        report.append("# Customer Segmentation Report (K-means Clustering)\n")
        report.append("## Executive Summary\n")
        report.append("This report presents a customer segmentation analysis based on K-means clustering of RFM (Recency, Frequency, Monetary) data.\n")
        
        report.append("## Overview of Customer Segments\n")
        
        segment_table = "| Segment ID | Segment Name | Customer Count | Percentage | Avg. Recency | Avg. Frequency | Avg. Monetary |\n"
        segment_table += "|------------|--------------|---------------|------------|-------------|---------------|---------------|\n"
        
        for segment_id, segment in self.segments.items():
            segment_table += f"| {segment['segment_id']} | {segment['segment_name']} | {segment['count']} | {segment['percentage']:.1f}% | {segment['avg_recency']:.1f} | {segment['avg_frequency']:.1f} | ${segment['avg_monetary']:.2f} |\n"
        
        report.append(segment_table)
        
        report.append("\n## Detailed Analysis of Each Segment\n")
        
        for segment_id, segment in self.segments.items():
            report.append(f"### {segment['segment_name']}\n")
            
            if segment_id in self.segment_descriptions:
                report.append(f"{self.segment_descriptions[segment_id]}\n")
            else:
                report.append(f"- Customer count: {segment['count']} ({segment['percentage']:.1f}%)\n")
                report.append(f"- Average recency: {segment['avg_recency']:.1f} days\n")
                report.append(f"- Average frequency: {segment['avg_frequency']:.1f} purchases\n")
                report.append(f"- Average monetary value: ${segment['avg_monetary']:.2f}\n")
            
            report.append("\n**Marketing Strategies:**\n")
            
            if segment_id in self.segment_strategies:
                report.append(f"{self.segment_strategies[segment_id]}\n")
            
            report.append("\n")
        
        report.append("## Conclusion\n")
        report.append("This segmentation using K-means clustering has identified distinct customer groups that can be targeted with specific marketing strategies.\n")
        
        return "\n".join(report)
    
    def plot_segments(self, output_file: str = "segment_visualization.png") -> str:
        """
        Generate a visualization of the customer segments
        """
        self.log("Generating segment visualization")
        
        if self.rfm_data is None or 'segment_name' not in self.rfm_data.columns:
            self.log("No segmentation results. Please perform segmentation first.")
            return ""
        
        # Create a figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Use a consistent color palette
        unique_segments = self.rfm_data['segment_name'].unique()
        color_palette = sns.color_palette('tab10', len(unique_segments))
        segment_colors = dict(zip(unique_segments, color_palette))
        
        # 1. Pie chart of segment sizes
        segment_counts = self.rfm_data['segment_name'].value_counts()
        axes[0, 0].pie(
            segment_counts, 
            labels=segment_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=[segment_colors[seg] for seg in segment_counts.index]
        )
        axes[0, 0].set_title('Customer Segment Distribution')
        
        # 2. Scatter plot of Recency vs Frequency
        sns.scatterplot(
            data=self.rfm_data,
            x='Recency',
            y='Frequency',
            hue='segment_name',
            palette=segment_colors,
            alpha=0.7,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Recency vs Frequency by Segment')
        
        # 3. Scatter plot of Frequency vs Monetary
        sns.scatterplot(
            data=self.rfm_data,
            x='Frequency',
            y='Monetary',
            hue='segment_name',
            palette=segment_colors,
            alpha=0.7,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Frequency vs Monetary by Segment')
        
        # 4. PCA visualization
        # Apply PCA for visualization
        X = self.rfm_data[['Recency', 'Frequency', 'Monetary']]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.scaler.transform(X))
        
        # Create a dataframe with PCA results
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['segment_name'] = self.rfm_data['segment_name'].values
        
        # Plot PCA results
        sns.scatterplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            hue='segment_name',
            palette=segment_colors,
            alpha=0.7,
            ax=axes[1, 1]
        )
        var_explained = pca.explained_variance_ratio_ * 100
        axes[1, 1].set_title(f'PCA Visualization (Explained Variance: {var_explained[0]:.1f}%, {var_explained[1]:.1f}%)')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.log(f"Visualization saved to {output_path}")
        return output_path
    
    def segment_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment a single customer based on their RFM data
        """
        self.log(f"Segmenting individual customer")
        
        if self.kmeans_model is None or self.scaler is None:
            self.log("No segmentation model. Please perform segmentation first.")
            return {"error": "No segmentation model available"}
        
        # Validate input
        required_fields = ['Recency', 'Frequency', 'Monetary']
        for field in required_fields:
            if field not in customer_data:
                self.log(f"Missing required field: {field}")
                return {"error": f"Missing required field: {field}"}
        
        # Prepare the customer data for prediction
        customer_array = np.array([[
            customer_data['Recency'],
            customer_data['Frequency'],
            customer_data['Monetary']
        ]])
        
        # Scale the data
        customer_scaled = self.scaler.transform(customer_array)
        
        # Predict the segment
        segment_id = int(self.kmeans_model.predict(customer_scaled)[0])
        segment_id_str = str(segment_id)
        
        # Get segment details
        if segment_id_str in self.segments:
            segment = self.segments[segment_id_str]
            
            result = {
                "segment_id": segment_id,
                "segment_name": segment["segment_name"],
                "description": self.segment_descriptions.get(segment_id_str, ""),
                "recommended_strategy": self.segment_strategies.get(segment_id_str, "")
            }
            return result
        else:
            self.log(f"Unexpected segment ID: {segment_id}")
            return {
                "segment_id": segment_id,
                "segment_name": f"Cluster {segment_id}",
                "description": "Unknown segment",
                "recommended_strategy": "Further analysis needed"
            }