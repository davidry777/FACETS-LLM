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
from agents.agent import Agent


class SegmentationAgent(Agent):
    """
    Agent that uses DeepSeek to segment customers based on RFM data
    """
    name = "Segmentation Agent"
    color = Agent.CYAN
    MODEL = "deepseek-r1:1.5b"
    
    def __init__(self):
        """
        Set up this instance by connecting to DeepSeek and initializing segmentation state
        """
        self.log("Initializing Segmentation Agent")
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Set up the DeepSeek client via Ollama
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        # Segmentation state
        self.rfm_data = None
        self.segments = None
        self.segment_descriptions = None
        self.segment_strategies = None
        
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
    
    def messages_for_segmentation_criteria(self, rfm_summary: Dict[str, Any], num_segments: int) -> List[Dict[str, str]]:
        """
        Create the message list for generating segmentation criteria
        """
        system_message = """You are an expert in customer segmentation and marketing analytics.
        Create segmentation criteria for an e-commerce business based on RFM (Recency, Frequency, Monetary) data.
        Respond with a JSON array of segment definitions with clear criteria for each."""
        
        user_message = f"""
        Here's a summary of the RFM data distribution:
        
        Recency (days since last purchase): 
        - Min: {rfm_summary["Recency"]["min"]}, Max: {rfm_summary["Recency"]["max"]}
        - Mean: {rfm_summary["Recency"]["mean"]}, Median: {rfm_summary["Recency"]["median"]}
        - 25th percentile: {rfm_summary["Recency"]["p25"]}, 75th percentile: {rfm_summary["Recency"]["p75"]}
        
        Frequency (number of purchases):
        - Min: {rfm_summary["Frequency"]["min"]}, Max: {rfm_summary["Frequency"]["max"]}
        - Mean: {rfm_summary["Frequency"]["mean"]}, Median: {rfm_summary["Frequency"]["median"]}
        - 25th percentile: {rfm_summary["Frequency"]["p25"]}, 75th percentile: {rfm_summary["Frequency"]["p75"]}
        
        Monetary (total spend):
        - Min: {rfm_summary["Monetary"]["min"]}, Max: {rfm_summary["Monetary"]["max"]}
        - Mean: {rfm_summary["Monetary"]["mean"]}, Median: {rfm_summary["Monetary"]["median"]}
        - 25th percentile: {rfm_summary["Monetary"]["p25"]}, 75th percentile: {rfm_summary["Monetary"]["p75"]}
        
        Correlations:
        - Recency-Frequency: {rfm_summary["correlations"]["Recency_Frequency"]}
        - Recency-Monetary: {rfm_summary["correlations"]["Recency_Monetary"]}
        - Frequency-Monetary: {rfm_summary["correlations"]["Frequency_Monetary"]}
        
        Please create {num_segments} customer segments with clear criteria for each. For each segment:
        1. Define specific ranges or thresholds for Recency, Frequency, and Monetary values
        2. Give the segment a descriptive name (like "Champions", "At Risk", etc.)
        3. Provide a brief description of this customer segment
        
        Format your response as a JSON array where each object has the following structure:
        [
            {{
                "segment_id": 1,
                "segment_name": "Champions",
                "criteria": {{
                    "recency": {{
                        "min": null,
                        "max": 30
                    }},
                    "frequency": {{
                        "min": 10,
                        "max": null
                    }},
                    "monetary": {{
                        "min": 1000,
                        "max": null
                    }}
                }},
                "description": "Recent customers who buy often and spend the most"
            }},
            ... (and so on)
        ]
        
        Return only the JSON array.
        """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "["}
        ]
    
    def messages_for_segment_insights(self, segments: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create the message list for generating segment insights
        """
        system_message = """You are an expert in marketing analytics and customer relationship management.
        Provide detailed descriptions and specific marketing strategies for customer segments.
        Respond with a JSON object containing detailed insights about each segment."""
        
        user_message = f"""
        I have customer segments based on RFM (Recency, Frequency, Monetary) analysis. For each segment, provide:
        1. A detailed description of the customer behavior and characteristics
        2. Specific marketing strategies and recommendations customized for this segment

        Here are the segments:
        {json.dumps(segments)}
        
        Format your response as a JSON object with two keys: "descriptions" and "strategies", each containing an object with segment IDs as keys:
        {{
            "descriptions": {{
                "1": "Detailed description for segment 1...",
                "2": "Detailed description for segment 2...",
                ...
            }},
            "strategies": {{
                "1": "Marketing strategies for segment 1...",
                "2": "Marketing strategies for segment 2...",
                ...
            }}
        }}
        
        Be specific and detailed in your recommendations. Include practical action items that a marketing team could implement.
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
        2. Introduction and Methodology
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
    
    def perform_segmentation(self, num_segments: int = 5) -> Dict[str, Any]:
        """
        Perform customer segmentation using DeepSeek LLM
        """
        self.log(f"Starting LLM-based segmentation with target of {num_segments} segments")
        
        if self.rfm_data is None:
            self.log("No data loaded. Please load data first.")
            return {}
        
        # Analyze RFM distribution
        rfm_summary = self._get_rfm_summary()
        
        # Step 1: Generate segmentation criteria using LLM
        self.log(f"Calling {self.MODEL} to generate segmentation criteria")
        
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for_segmentation_criteria(rfm_summary, num_segments),
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            segmentation_criteria = self.extract_json(content)
            
            if not segmentation_criteria:
                self.log("Failed to parse segmentation criteria from LLM response")
                segmentation_criteria = self._get_fallback_criteria(num_segments)
        except Exception as e:
            self.log(f"Error using LLM for segmentation criteria: {str(e)}")
            segmentation_criteria = self._get_fallback_criteria(num_segments)
        
        # Step 2: Apply the criteria to segment customers
        self.segments = self._apply_segmentation_criteria(segmentation_criteria)
        
        # Step 3: Generate segment descriptions and strategies
        try:
            self.log(f"Calling {self.MODEL} to generate segment insights")
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for_segment_insights(self.segments),
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            segment_insights = self.extract_json(content)
            
            if not segment_insights or "descriptions" not in segment_insights or "strategies" not in segment_insights:
                self.log("Failed to parse segment insights from LLM response")
                insights = self._get_fallback_insights(self.segments)
                self.segment_descriptions = insights["descriptions"]
                self.segment_strategies = insights["strategies"]
            else:
                self.segment_descriptions = segment_insights["descriptions"]
                self.segment_strategies = segment_insights["strategies"]
        except Exception as e:
            self.log(f"Error using LLM for segment insights: {str(e)}")
            insights = self._get_fallback_insights(self.segments)
            self.segment_descriptions = insights["descriptions"]
            self.segment_strategies = insights["strategies"]
        
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
        with open("output/llm_segmentation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.log(f"Segmentation complete - created {len(self.segments)} segments")
        return results
    
    def _apply_segmentation_criteria(self, criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply the segmentation criteria to the customers"""
        self.log("Applying segmentation criteria to customers")
        
        # Initialize segment assignments
        self.rfm_data['segment_id'] = None
        self.rfm_data['segment_name'] = None
        
        # Apply each segment's criteria in order
        for segment in criteria:
            segment_id = segment["segment_id"]
            segment_name = segment["segment_name"]
            
            # Create a mask for this segment
            mask = pd.Series(True, index=self.rfm_data.index)
            
            # Apply Recency criteria
            if segment["criteria"]["recency"]["min"] is not None:
                mask &= self.rfm_data['Recency'] >= segment["criteria"]["recency"]["min"]
            if segment["criteria"]["recency"]["max"] is not None:
                mask &= self.rfm_data['Recency'] <= segment["criteria"]["recency"]["max"]
            
            # Apply Frequency criteria
            if segment["criteria"]["frequency"]["min"] is not None:
                mask &= self.rfm_data['Frequency'] >= segment["criteria"]["frequency"]["min"]
            if segment["criteria"]["frequency"]["max"] is not None:
                mask &= self.rfm_data['Frequency'] <= segment["criteria"]["frequency"]["max"]
            
            # Apply Monetary criteria
            if segment["criteria"]["monetary"]["min"] is not None:
                mask &= self.rfm_data['Monetary'] >= segment["criteria"]["monetary"]["min"]
            if segment["criteria"]["monetary"]["max"] is not None:
                mask &= self.rfm_data['Monetary'] <= segment["criteria"]["monetary"]["max"]
            
            # Assign customers to this segment (only if not already assigned)
            unassigned_mask = self.rfm_data['segment_id'].isna()
            assignment_mask = mask & unassigned_mask
            
            self.rfm_data.loc[assignment_mask, 'segment_id'] = segment_id
            self.rfm_data.loc[assignment_mask, 'segment_name'] = segment_name
        
        # Handle any unassigned customers by putting them in an "Other" segment
        unassigned_mask = self.rfm_data['segment_id'].isna()
        if unassigned_mask.sum() > 0:
            next_id = max([s["segment_id"] for s in criteria]) + 1
            self.rfm_data.loc[unassigned_mask, 'segment_id'] = next_id
            self.rfm_data.loc[unassigned_mask, 'segment_name'] = "Other"
            
            # Add the "Other" segment to criteria
            criteria.append({
                "segment_id": next_id,
                "segment_name": "Other",
                "criteria": {
                    "recency": {"min": None, "max": None},
                    "frequency": {"min": None, "max": None},
                    "monetary": {"min": None, "max": None}
                },
                "description": "Customers who didn't fit into any defined segment"
            })
        
        # Create segments summary
        segments = {}
        for segment in criteria:
            segment_id = segment["segment_id"]
            segment_mask = self.rfm_data['segment_id'] == segment_id
            segment_data = self.rfm_data[segment_mask]
            
            segments[str(segment_id)] = {
                "segment_id": segment_id,
                "segment_name": segment["segment_name"],
                "count": int(segment_mask.sum()),
                "percentage": float(segment_mask.sum() / len(self.rfm_data) * 100),
                "criteria": segment["criteria"],
                "description": segment["description"],
                "avg_recency": float(segment_data['Recency'].mean() if len(segment_data) > 0 else 0),
                "avg_frequency": float(segment_data['Frequency'].mean() if len(segment_data) > 0 else 0),
                "avg_monetary": float(segment_data['Monetary'].mean() if len(segment_data) > 0 else 0),
                "total_monetary": float(segment_data['Monetary'].sum() if len(segment_data) > 0 else 0)
            }
        
        return segments
    
    def _get_fallback_criteria(self, num_segments: int) -> List[Dict[str, Any]]:
        """Generate fallback segmentation criteria if LLM fails"""
        # Calculate percentiles for simple segmentation
        recency_thresholds = [
            self.rfm_data['Recency'].quantile(q) 
            for q in np.linspace(0, 1, num_segments + 1)[1:-1]
        ]
        frequency_thresholds = [
            self.rfm_data['Frequency'].quantile(q) 
            for q in np.linspace(0, 1, num_segments + 1)[1:-1]
        ]
        monetary_thresholds = [
            self.rfm_data['Monetary'].quantile(q) 
            for q in np.linspace(0, 1, num_segments + 1)[1:-1]
        ]
        
        # Create simple criteria
        if num_segments <= 3:
            # Create RFM-based segments
            criteria = [
                {
                    "segment_id": 1,
                    "segment_name": "Champions",
                    "criteria": {
                        "recency": {"min": None, "max": recency_thresholds[0]},
                        "frequency": {"min": frequency_thresholds[-1], "max": None},
                        "monetary": {"min": monetary_thresholds[-1], "max": None}
                    },
                    "description": "Recent customers who buy often and spend the most"
                },
                {
                    "segment_id": 2,
                    "segment_name": "Potential Loyalists",
                    "criteria": {
                        "recency": {"min": None, "max": recency_thresholds[0]},
                        "frequency": {"min": None, "max": frequency_thresholds[-1]},
                        "monetary": {"min": None, "max": monetary_thresholds[-1]}
                    },
                    "description": "Recent customers with moderate frequency and monetary value"
                },
                {
                    "segment_id": 3,
                    "segment_name": "At Risk",
                    "criteria": {
                        "recency": {"min": recency_thresholds[0], "max": None},
                        "frequency": {"min": frequency_thresholds[-1], "max": None},
                        "monetary": {"min": monetary_thresholds[-1], "max": None}
                    },
                    "description": "Customers who used to purchase frequently and spend a lot but haven't purchased recently"
                }
            ]
        else:
            # Create more segments if needed
            criteria = []
            for i in range(num_segments):
                criteria.append({
                    "segment_id": i + 1,
                    "segment_name": f"Segment {i + 1}",
                    "criteria": {"recency": {"min": None, "max": None}, "frequency": {"min": None, "max": None}, "monetary": {"min": None, "max": None}},
                    "description": f"Automatically generated segment {i + 1}"
                })
        
        return criteria[:num_segments]
    
    def _get_fallback_insights(self, segments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback insights if LLM fails"""
        descriptions = {}
        strategies = {}
        
        for segment_id, segment in segments.items():
            name = segment["segment_name"]
            
            # Create basic descriptions
            descriptions[segment_id] = (
                f"{name}: This segment contains {segment['count']} customers ({segment['percentage']:.1f}% of total). "
                f"On average, they made their last purchase {segment['avg_recency']:.1f} days ago, "
                f"have made {segment['avg_frequency']:.1f} purchases, and "
                f"spent ${segment['avg_monetary']:.2f} per customer."
            )
            
            # Create basic strategies
            if "Champions" in name or "Loyal" in name:
                strategies[segment_id] = (
                    "Reward these customers with loyalty programs, exclusive offers, and early access to new products. "
                    "Focus on building relationships and turning them into brand advocates."
                )
            elif "Risk" in name or "Dormant" in name:
                strategies[segment_id] = (
                    "Implement a win-back campaign with personalized offers based on past purchases. "
                    "Send reminders about your products and why they chose you before."
                )
            elif "New" in name:
                strategies[segment_id] = (
                    "Focus on onboarding and education about your products and services. "
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
        output_path = os.path.join("output", output_file)
        with open(output_path, "w") as f:
            f.write(report_content)
        
        self.log(f"Report saved to {output_path}")
        return output_path
    
    def _generate_fallback_report(self) -> str:
        """Generate a simple fallback report if LLM fails"""
        report = []
        report.append("# Customer Segmentation Report\n")
        report.append("## Executive Summary\n")
        report.append("This report presents a customer segmentation analysis based on RFM (Recency, Frequency, Monetary) data.\n")
        
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
        report.append("This segmentation can be used to develop targeted marketing strategies for each customer group.\n")
        
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
        
        # 1. Pie chart of segment sizes
        segment_counts = self.rfm_data['segment_name'].value_counts()
        axes[0, 0].pie(
            segment_counts, 
            labels=segment_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=sns.color_palette('tab10', len(segment_counts))
        )
        axes[0, 0].set_title('Customer Segment Distribution')
        
        # 2. Scatter plot of Recency vs Frequency
        sns.scatterplot(
            data=self.rfm_data,
            x='Recency',
            y='Frequency',
            hue='segment_name',
            palette='tab10',
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
            palette='tab10',
            alpha=0.7,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Frequency vs Monetary by Segment')
        
        # 4. Bar chart of average metrics by segment
        segment_avgs = self.rfm_data.groupby('segment_name')[['Recency', 'Frequency', 'Monetary']].mean()
        
        # Normalize the metrics for better visualization
        segment_avgs_norm = segment_avgs.copy()
        for col in segment_avgs_norm.columns:
            if col == 'Recency':
                # For Recency, lower is better, so invert the normalization
                segment_avgs_norm[col] = 1 - (segment_avgs_norm[col] / segment_avgs_norm[col].max())
            else:
                segment_avgs_norm[col] = segment_avgs_norm[col] / segment_avgs_norm[col].max()
        
        segment_avgs_norm.plot(
            kind='bar',
            ax=axes[1, 1],
            color=['#ff9999', '#66b3ff', '#99ff99'],
            rot=45
        )
        axes[1, 1].set_title('Normalized RFM Metrics by Segment')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend(title='Metric')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join("output", output_file)
        plt.savefig(output_path)
        plt.close()
        
        self.log(f"Visualization saved to {output_path}")
        return output_path
    
    def segment_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment a single customer based on their RFM data
        """
        self.log(f"Segmenting individual customer")
        
        if self.segments is None:
            self.log("No segmentation model. Please perform segmentation first.")
            return {"error": "No segmentation model available"}
        
        # Validate input
        required_fields = ['Recency', 'Frequency', 'Monetary']
        for field in required_fields:
            if field not in customer_data:
                self.log(f"Missing required field: {field}")
                return {"error": f"Missing required field: {field}"}
        
        # Check each segment's criteria to find a match
        for segment_id, segment in self.segments.items():
            criteria = segment["criteria"]
            
            # Check if customer matches this segment's criteria
            recency_match = True
            if criteria["recency"]["min"] is not None:
                recency_match = recency_match and customer_data['Recency'] >= criteria["recency"]["min"]
            if criteria["recency"]["max"] is not None:
                recency_match = recency_match and customer_data['Recency'] <= criteria["recency"]["max"]
                
            frequency_match = True
            if criteria["frequency"]["min"] is not None:
                frequency_match = frequency_match and customer_data['Frequency'] >= criteria["frequency"]["min"]
            if criteria["frequency"]["max"] is not None:
                frequency_match = frequency_match and customer_data['Frequency'] <= criteria["frequency"]["max"]
                
            monetary_match = True
            if criteria["monetary"]["min"] is not None:
                monetary_match = monetary_match and customer_data['Monetary'] >= criteria["monetary"]["min"]
            if criteria["monetary"]["max"] is not None:
                monetary_match = monetary_match and customer_data['Monetary'] <= criteria["monetary"]["max"]
            
            # If all criteria match, return this segment
            if recency_match and frequency_match and monetary_match:
                result = {
                    "segment_id": int(segment_id),
                    "segment_name": segment["segment_name"],
                    "description": segment["description"],
                    "recommended_strategy": self.segment_strategies.get(segment_id, "")
                }
                return result
        
        # If no match found, return the "Other" segment
        self.log("No exact segment match found, returning 'Other' segment")
        return {
            "segment_id": 0,
            "segment_name": "Other",
            "description": "This customer doesn't fit into any defined segment",
            "recommended_strategy": "Conduct further analysis to understand this customer's behavior"
        }