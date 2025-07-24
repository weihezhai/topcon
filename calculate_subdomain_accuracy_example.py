import json
from calculate_subdomain_accuracy import calculate_subdomain_accuracy, print_results, save_results

# Create example data files for testing
def create_example_data():
    # Example predictions
    predictions = {
        "index0": {
            "yes": 0.0035936026833951473,
            "no": 0.9964063763618469,
            "yes_no_diff": -0.9928127736784518,
            "prediction": "no",
            "label": 0,
            "correctness": True
        },
        "index1": {
            "yes": 0.00037998449988663197,
            "no": 0.9996199607849121,
            "yes_no_diff": -0.9992399762850255,
            "prediction": "no",
            "label": 0,
            "correctness": True
        },
        "index2": {
            "yes": 0.000552778597921133,
            "no": 0.9994471669197083,
            "yes_no_diff": -0.9988943883217871,
            "prediction": "no",
            "label": 1,
            "correctness": False
        }
    }
    
    # Example index to paper_id mapping
    index_to_paper_id = {
        "0": "0A6f1b66pE",
        "1": "0GzqVqCKns",
        "2": "0iAZYF9hrl"
    }
    
    # Example subdomain paper ids (simplified version)
    subdomain_paper_ids = {
        "diffusion": ["0GzqVqCKns", "0A6f1b66pE"],
        "video": ["0iAZYF9hrl", "0GzqVqCKns"],
        "3d": ["0A6f1b66pE"]
    }
    
    # Save example data
    with open("example_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    
    with open("example_index_to_paper_id.json", "w") as f:
        json.dump(index_to_paper_id, f, indent=2)
    
    with open("example_subdomain_paper_ids.json", "w") as f:
        json.dump(subdomain_paper_ids, f, indent=2)
    
    return "example_predictions.json", "example_index_to_paper_id.json", "example_subdomain_paper_ids.json"

if __name__ == "__main__":
    # Create example data
    pred_path, index_path, subdomain_path = create_example_data()
    
    # Calculate accuracies
    results = calculate_subdomain_accuracy(pred_path, index_path, subdomain_path)
    
    # Print and save results
    print_results(results)
    save_results(results, "example_subdomain_accuracy_results.json")
    
    # Clean up example files (optional)
    import os
    os.remove(pred_path)
    os.remove(index_path)
    os.remove(subdomain_path)
