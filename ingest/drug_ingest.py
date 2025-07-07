import os
import json
from fda_ingest import load_txt_files

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def debug_table_structure():
    """Debug function to show table structures"""
    tables = load_txt_files()
    
    print("\n=== TABLE STRUCTURES ===")
    for name, df in tables.items():
        print(f"\n{name}: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"Sample data:\n{df.head(2)}")
    
    return tables

def create_documents():
    tables = load_txt_files()

    applications = tables.get("Applications")
    products = tables.get("Products")
    submissions = tables.get("Submissions")
    marketing_status = tables.get("MarketingStatus")
    marketing_status_lookup = tables.get("MarketingStatus_Lookup")
    te = tables.get("TE")  # This table has MarketingStatusID linkage

    if applications is None or products is None:
        raise ValueError("Missing core tables")

    print(f"\n=== EFFICIENT MERGING STRATEGY ===")
    print(f"Products: {products.shape}")
    print(f"Applications: {applications.shape}")
    
    # Start with Products (core table)
    merged = products.copy()
    print(f"Starting with Products: {merged.shape}")
    
    # Merge with Applications
    merged = merged.merge(applications, on="ApplNo", how="left")
    print(f"After Applications merge: {merged.shape}")
    
    # Add marketing status from TE table (more direct)
    if te is not None:
        print(f"TE table: {te.shape}")
        merged = merged.merge(te[['ApplNo', 'ProductNo', 'MarketingStatusID', 'TECode']], 
                            on=['ApplNo', 'ProductNo'], how="left")
        print(f"After TE merge: {merged.shape}")
        
        # Add marketing status descriptions
        if marketing_status_lookup is not None:
            merged = merged.merge(marketing_status_lookup, on="MarketingStatusID", how="left")
            print(f"After MarketingStatus_Lookup merge: {merged.shape}")
    
    # Get latest submission info (aggregate to avoid explosion)
    if submissions is not None:
        print(f"Aggregating submissions data...")
        # Only aggregate the most recent submission per application
        latest_submissions = submissions.sort_values('SubmissionStatusDate', na_position='first').groupby('ApplNo').agg({
            'SubmissionType': 'last',
            'SubmissionStatus': 'last', 
            'ReviewPriority': 'last'
        }).reset_index()
        
        merged = merged.merge(latest_submissions, on="ApplNo", how="left")
        print(f"After aggregated submissions merge: {merged.shape}")

    # Clean up column names and create documents
    print(f"\n=== CREATING DOCUMENTS ===")
    docs = []
    for _, row in merged.iterrows():
        # Handle missing values gracefully
        drug_name = str(row.get("DrugName", "") or "")
        form = str(row.get("Form", "") or "")
        strength = str(row.get("Strength", "") or "")
        active_ingredient = str(row.get("ActiveIngredient", "") or "")
        
        doc = {
            "drug_name": drug_name,
            "application_no": str(row.get("ApplNo", "")),
            "product_no": str(row.get("ProductNo", "")),
            "form": form,
            "strength": strength,
            "active_ingredient": active_ingredient,
            "marketing_status": str(row.get("MarketingStatusDescription", "") or ""),
            "submission_type": str(row.get("SubmissionType", "") or ""),
            "submission_status": str(row.get("SubmissionStatus", "") or ""),
            "te_code": str(row.get("TECode", "") or ""),
            "sponsor_name": str(row.get("SponsorName", "") or ""),
            "application_type": str(row.get("ApplType", "") or ""),
            "description": f"{drug_name} ({active_ingredient}) is a {form} formulation with strength {strength}." if drug_name else ""
        }
        docs.append(doc)

    # Save each doc as JSONL
    with open(os.path.join(OUTPUT_DIR, "fda_documents.jsonl"), "w", encoding="utf-8") as f:
        for doc in docs:
            json.dump(doc, f)
            f.write("\n")

    print(f"✅ Saved {len(docs)} documents to {OUTPUT_DIR}/fda_documents.jsonl")

if __name__ == "__main__":
    create_documents()