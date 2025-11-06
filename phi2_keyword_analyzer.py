"""
Google Ads Keyword Analyzer using Microsoft Phi-2 (FREE Model)
Analyzes keywords and identifies those with conversions > 25
"""

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
import os
warnings.filterwarnings('ignore')

class GoogleAdsKeywordAnalyzer:
    def __init__(self):
        """Initialize the analyzer with Microsoft Phi-2 model"""
        print("="*70)
        print("🚀 GOOGLE ADS KEYWORD ANALYZER")
        print("   Powered by Microsoft Phi-2 (FREE Model)")
        print("="*70)
        print("\n📥 Loading Microsoft Phi-2 model...")
        print("   (This may take a few minutes on first run - model will be cached)")
        
        # Load the FREE Phi-2 model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!\n")
        self.threshold = 25  # 25 conversions threshold (NOT conversion rate)
    
    def load_data(self, csv_file):
        """Load Google Ads data from CSV file"""
        print("📊 Loading Google Ads data...")
        
        # Check if file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ File not found: {csv_file}")
        
        # Load CSV data from file
        self.df = pd.read_csv(csv_file)
        
        # Clean conversion rate column (keep for display)
        self.df['Conv. rate'] = self.df['Conv. rate'].str.replace('%', '').astype(float)
        
        print(f"✅ Loaded {len(self.df)} keywords from {csv_file}\n")
        return self.df
    
    def analyze_with_ai(self):
        """Use Phi-2 to analyze the data and provide insights"""
        print("="*70)
        print("🤖 AI ANALYSIS WITH PHI-2")
        print("="*70)
        
        # Prepare summary of data for AI
        summary = f"""Analyze this Google Ads data:
- Total keywords: {len(self.df)}
- Highest conversions: {self.df['Conversions'].max():.0f}
- Average conversions: {self.df['Conversions'].mean():.2f}
- Threshold for ads: {self.threshold} conversions

Top 3 keywords by conversions:
"""
        top_3 = self.df.nlargest(3, 'Conversions')
        for idx, row in top_3.iterrows():
            summary += f"- {row['Keyword']}: {row['Conversions']:.0f} conversions\n"
        
        summary += f"\nQuestion: How many keywords have conversions >= {self.threshold}?"
        
        # Generate AI response
        print("\n🧠 AI is analyzing the data...\n")
        inputs = self.tokenizer(summary, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_length=300, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.batch_decode(outputs)[0]
        
        print("💬 AI Response:")
        print("-" * 70)
        print(response)
        print("-" * 70)
        print()
    
    def filter_keywords(self):
        """Filter keywords based on 25 conversions threshold"""
        print("="*70)
        print(f"🔍 FILTERING KEYWORDS (Conversions >= {self.threshold})")
        print("="*70)
        
        # Filter keywords with conversions >= 25
        eligible_keywords = self.df[self.df['Conversions'] >= self.threshold]
        
        print(f"\n📊 RESULTS:")
        print(f"   Total keywords analyzed: {len(self.df)}")
        print(f"   Keywords with Conversions >= {self.threshold}: {len(eligible_keywords)}")
        print(f"   Keywords with Conversions < {self.threshold}: {len(self.df) - len(eligible_keywords)}")
        
        if len(eligible_keywords) == 0:
            print(f"\n❌ NO KEYWORDS MEET THE {self.threshold} CONVERSIONS THRESHOLD!")
            print(f"\n⚠️  IMPORTANT FINDING:")
            print(f"   - Highest conversions in your data: {self.df['Conversions'].max():.0f}")
            print(f"   - Average conversions: {self.df['Conversions'].mean():.2f}")
            print(f"\n💡 Recommendation: Lower threshold to 15-20 conversions")
            
            print(f"\n📋 TOP 5 PERFORMING KEYWORDS (for reference):")
            print("=" * 70)
            top_5 = self.df.nlargest(5, 'Conversions')
            for idx, row in top_5.iterrows():
                print(f"\n   Keyword: {row['Keyword']}")
                print(f"   Campaign: {row['Campaign']}")
                print(f"   Conversions: {int(row['Conversions'])}")
                print(f"   Conversion Rate: {row['Conv. rate']:.2f}%")
                print(f"   Status: {row['Status']}")
        else:
            print(f"\n✅ ELIGIBLE KEYWORDS FOR ADS:")
            print("=" * 70)
            for idx, row in eligible_keywords.iterrows():
                print(f"\n   ✓ Keyword: {row['Keyword']}")
                print(f"     Campaign: {row['Campaign']}")
                print(f"     Ad Group: {row['Ad group']}")
                print(f"     Conversions: {int(row['Conversions'])}")
                print(f"     Conversion Rate: {row['Conv. rate']:.2f}%")
                print(f"     Cost: {row['Cost']}")
                print(f"     Match Type: {row['Match type']}")
                print(f"     Status: {row['Status']}")
        
        return eligible_keywords
    
    def generate_keyword_list(self, eligible_keywords):
        """Generate final list of keywords for ads"""
        print("\n" + "="*70)
        print("📝 FINAL KEYWORD LIST FOR ADS")
        print("="*70)
        
        if len(eligible_keywords) == 0:
            print(f"\n❌ No keywords meet the {self.threshold} conversions threshold.")
            print("\n💡 ALTERNATIVE APPROACH:")
            print("   Since no keywords meet the threshold, here are your BEST performers:")
            print("\n   TOP 10 KEYWORDS (use these for your ads):")
            print("   " + "-" * 66)
            
            best_keywords = self.df.nlargest(10, 'Conversions')
            keyword_list = []
            
            for rank, (idx, row) in enumerate(best_keywords.iterrows(), 1):
                keyword_info = {
                    'rank': rank,
                    'keyword': row['Keyword'],
                    'conversions': int(row['Conversions']),
                    'conversion_rate': f"{row['Conv. rate']:.2f}%",
                    'campaign': row['Campaign'],
                    'status': row['Status']
                }
                keyword_list.append(keyword_info)
                
                print(f"\n   {rank}. {row['Keyword']}")
                print(f"      Conversions: {int(row['Conversions'])} | Campaign: {row['Campaign']}")
            
            # Return only Keyword and Conversions columns
            df_to_save = best_keywords[['Keyword', 'Conversions']].copy()
            return keyword_list, df_to_save
        else:
            print(f"\n✅ Keywords eligible for ads (Conversions >= {self.threshold}):\n")
            keyword_list = []
            
            for idx, row in eligible_keywords.iterrows():
                keyword_info = {
                    'keyword': row['Keyword'],
                    'conversions': int(row['Conversions']),
                    'conversion_rate': f"{row['Conv. rate']:.2f}%",
                    'campaign': row['Campaign'],
                    'status': row['Status']
                }
                keyword_list.append(keyword_info)
                
                print(f"   ✓ {row['Keyword']} ({int(row['Conversions'])} conversions)")
            
            # Return only Keyword and Conversions columns
            df_to_save = eligible_keywords[['Keyword', 'Conversions']].copy()
            return keyword_list, df_to_save
    
    def run_analysis(self, csv_file):
        """Run complete analysis pipeline"""
        # Load data
        self.load_data(csv_file)
        
        # AI analysis
        self.analyze_with_ai()
        
        # Filter keywords
        eligible_keywords = self.filter_keywords()
        
        # Generate final list (with only Keyword and Conversions columns)
        keyword_list, df_to_save = self.generate_keyword_list(eligible_keywords)
        
        # ⭐ SAVE TO CSV (Only Keyword and Conversions columns)
        if len(eligible_keywords) > 0:
            output_file = 'eligible_keywords_conversions_25plus.csv'
            df_to_save.to_csv(output_file, index=False)
            print(f"\n💾 ✅ Eligible keywords saved to: {output_file}")
            print(f"📋 Columns: Keyword, Conversions")
        else:
            output_file = 'top_10_keywords_by_conversions.csv'
            df_to_save.to_csv(output_file, index=False)
            print(f"\n💾 ✅ Top 10 keywords saved to: {output_file}")
            print(f"📋 Columns: Keyword, Conversions")
        
        # Final summary
        print("\n" + "="*70)
        print("📊 ANALYSIS COMPLETE")
        print("="*70)
        print(f"✅ Analyzed {len(self.df)} keywords")
        print(f"✅ Found {len(eligible_keywords)} keywords with Conversions >= {self.threshold}")
        if len(eligible_keywords) == 0:
            print(f"💡 Showing top 10 performers as alternative")
        print(f"✅ Results saved to: {output_file}")
        print("="*70 + "\n")
        
        return keyword_list


# Main execution
if __name__ == "__main__":
    # CSV file name
    csv_file = "google_ads_campaign_data.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found!")
        print(f"📂 Please make sure the file is in the same directory as this script.")
        exit(1)
    
    # Initialize analyzer
    analyzer = GoogleAdsKeywordAnalyzer()
    
    # Run analysis
    results = analyzer.run_analysis(csv_file)
    
    print("\n✅ Analysis complete!")
    print(f"📋 Total keywords for ads: {len(results)}")