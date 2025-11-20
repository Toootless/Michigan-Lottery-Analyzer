#!/usr/bin/env python3
"""
CSV Data Integration System
Update the main lottery application to use enhanced CSV data from OCR processing
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataIntegrator:
    """Integrate enhanced CSV data into the main application"""
    
    def __init__(self):
        """Initialize the integrator"""
        self.enhanced_dir = Path("enhanced_data")
        self.past_games_dir = Path("past_games")
        
        # Mapping of enhanced files to production files
        self.file_mapping = {
            'Powerball_Enhanced.csv': 'Powerball numbers from LotteryUSA.csv',
            'Mega_Millions_Enhanced.csv': 'Mega Millions numbers from LotteryUSA.csv',
            'Lucky_for_Life_Enhanced.csv': 'Lucky for Life numbers from LotteryUSA.csv',
            'Daily_3_Enhanced.csv': 'MI Daily 3 Evening numbers from LotteryUSA.csv',
            'Daily_4_Enhanced.csv': 'MI Daily 4 Evening numbers from LotteryUSA.csv',
            'Lotto_47_Enhanced.csv': 'MI Lotto 47 numbers from LotteryUSA.csv'
        }
    
    def backup_existing_files(self):
        """Backup existing CSV files"""
        backup_dir = self.past_games_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        logger.info("💾 Creating backups of existing CSV files...")
        
        backed_up = []
        for enhanced_file, production_file in self.file_mapping.items():
            production_path = self.past_games_dir / production_file
            
            if production_path.exists():
                backup_path = backup_dir / f"{production_file}.backup"
                shutil.copy2(production_path, backup_path)
                backed_up.append(production_file)
                logger.info(f"   ✅ Backed up: {production_file}")
        
        logger.info(f"📁 Backed up {len(backed_up)} files to {backup_dir}")
        return len(backed_up)
    
    def integrate_enhanced_data(self):
        """Replace production CSV files with enhanced versions"""
        
        if not self.enhanced_dir.exists():
            logger.error(f"❌ Enhanced data directory not found: {self.enhanced_dir}")
            return False
        
        logger.info("🔄 Integrating enhanced CSV data into production...")
        
        integrated = []
        
        for enhanced_file, production_file in self.file_mapping.items():
            enhanced_path = self.enhanced_dir / enhanced_file
            production_path = self.past_games_dir / production_file
            
            if enhanced_path.exists():
                # Copy enhanced file to production location
                shutil.copy2(enhanced_path, production_path)
                integrated.append(production_file)
                logger.info(f"   ✅ Updated: {production_file}")
            else:
                logger.warning(f"   ⚠️ Enhanced file not found: {enhanced_file}")
        
        logger.info(f"🎯 Integrated {len(integrated)} enhanced CSV files")
        return len(integrated) > 0
    
    def verify_integration(self):
        """Verify that the integration was successful"""
        logger.info("🔍 Verifying integration...")
        
        verification_results = {}
        
        for enhanced_file, production_file in self.file_mapping.items():
            production_path = self.past_games_dir / production_file
            
            if production_path.exists():
                try:
                    # Count lines in the file
                    with open(production_path, 'r') as f:
                        line_count = sum(1 for line in f) - 1  # Subtract header
                    
                    verification_results[production_file] = line_count
                    logger.info(f"   ✅ {production_file}: {line_count} records")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error reading {production_file}: {e}")
                    verification_results[production_file] = 0
            else:
                logger.warning(f"   ⚠️ File not found: {production_file}")
                verification_results[production_file] = 0
        
        return verification_results
    
    def generate_summary_report(self, verification_results: dict):
        """Generate a summary report of the integration"""
        
        report = []
        report.append("="*60)
        report.append("📊 CSV DATA INTEGRATION SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        total_records = 0
        
        # Game-specific summaries
        game_summaries = {
            'Powerball numbers from LotteryUSA.csv': 'Powerball',
            'Mega Millions numbers from LotteryUSA.csv': 'Mega Millions',
            'Lucky for Life numbers from LotteryUSA.csv': 'Lucky for Life',
            'MI Daily 3 Evening numbers from LotteryUSA.csv': 'Daily 3',
            'MI Daily 4 Evening numbers from LotteryUSA.csv': 'Daily 4',
            'MI Lotto 47 numbers from LotteryUSA.csv': 'Lotto 47'
        }
        
        for file_name, game_name in game_summaries.items():
            record_count = verification_results.get(file_name, 0)
            report.append(f"🎯 {game_name:15} : {record_count:,} records")
            total_records += record_count
        
        report.append("")
        report.append(f"📈 TOTAL RECORDS    : {total_records:,}")
        report.append("")
        
        # Data sources
        report.append("📥 DATA SOURCES:")
        report.append("   • LotteryUSA.com (recent data)")
        report.append("   • PDF OCR extraction (historical data 2019-2021)")
        report.append("   • Data validation and cleaning applied")
        report.append("")
        
        # Benefits
        report.append("✅ ENHANCEMENTS:")
        report.append("   • Significantly expanded historical data")
        report.append("   • OCR-extracted data from 3 years of PDFs")
        report.append("   • Data validation and cleaning applied")
        report.append("   • Duplicate removal and date standardization")
        report.append("   • Ready for faster CSV-based processing")
        report.append("")
        
        report.append("🚀 The main lottery application now has access to")
        report.append(f"   enhanced datasets with {total_records:,} total records!")
        report.append("="*60)
        
        return "\n".join(report)
    
    def run_integration(self):
        """Run the complete integration process"""
        
        print("🔄 CSV Data Integration System")
        print("=" * 40)
        print()
        
        # Step 1: Backup existing files
        backup_count = self.backup_existing_files()
        print()
        
        # Step 2: Integrate enhanced data
        success = self.integrate_enhanced_data()
        print()
        
        if not success:
            logger.error("❌ Integration failed!")
            return False
        
        # Step 3: Verify integration
        verification_results = self.verify_integration()
        print()
        
        # Step 4: Generate summary report
        summary_report = self.generate_summary_report(verification_results)
        print(summary_report)
        
        # Save report to file
        report_file = Path("CSV_Integration_Report.txt")
        with open(report_file, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"📄 Summary report saved to: {report_file}")
        
        return True

def main():
    """Main function"""
    integrator = CSVDataIntegrator()
    success = integrator.run_integration()
    
    if success:
        print("\n🎉 CSV data integration completed successfully!")
        print("   The main lottery application now uses enhanced data.")
    else:
        print("\n❌ CSV data integration failed!")

if __name__ == "__main__":
    main()