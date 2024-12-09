import os
import json
import logging
from datetime import datetime
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar

def setup_test_logging(log_dir="test_logs"):
    """Set up test logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mp_test_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_available_potcar_types(element, potcar_dir):
    """Get available POTCAR types for a given element"""
    element_folders = []
    
    # Scan all folders in the directory
    for folder in os.listdir(potcar_dir):
        # Remove all suffixes (_GW, _sv_GW, etc.) to get the base element name
        base_element = folder.split('_')[0]
        
        # Check if it is the element we are looking for
        if base_element == element:
            potcar_path = os.path.join(potcar_dir, folder, 'POTCAR')
            if os.path.exists(potcar_path):
                element_folders.append(folder)
    
    return sorted(element_folders)  # Sort to ensure consistency

def select_potcar_type(element, available_types, preferred_types=None):
    """Select POTCAR type based on priority"""
    if not available_types:
        return None
        
    if preferred_types and element in preferred_types:
        if preferred_types[element] in available_types:
            return preferred_types[element]
    
    # Priority order
    priority_suffixes = [
        '_pv',         # With p valence electrons
        '_sv',         # With semi-core valence electrons
        '',           # Standard version
        '_GW',        # GW version
        '_sv_GW',     # GW with semi-core valence electrons
        '_h',         # Hard version
        '_s'          # Soft version
    ]
    
    # Search by priority
    for suffix in priority_suffixes:
        for pot_type in available_types:
            # Check if it matches the current priority
            if pot_type == f"{element}{suffix}":
                return pot_type
            
    # If no preferred type is found, return the first available type
    return available_types[0]

def combine_potcar_files(element_list, potcar_dir, logger, preferred_types=None):
    """Combine POTCAR files"""
    logger.info(f"Combining POTCAR files for elements: {element_list}")
    
    # Record available pseudopotential types
    available_types = {}
    for element in element_list:
        types = get_available_potcar_types(element, potcar_dir)
        available_types[element] = types
        logger.info(f"Available POTCAR types for {element}: {types}")
    
    combined_content = []
    selected_types = {}
    missing_elements = []
    
    for element in element_list:
        if not available_types[element]:
            logger.error(f"No POTCAR found for element {element}")
            missing_elements.append(element)
            continue
            
        selected_type = select_potcar_type(element, available_types[element], preferred_types)
        selected_types[element] = selected_type
        
        potcar_path = os.path.join(potcar_dir, selected_type, 'POTCAR')
        logger.info(f"Using POTCAR type '{selected_type}' for element {element}")
        
        try:
            with open(potcar_path, 'r') as f:
                content = f.read()
                combined_content.append(content)
        except Exception as e:
            logger.error(f"Error reading POTCAR for {selected_type}: {str(e)}")
            missing_elements.append(element)
    
    if missing_elements:
        raise FileNotFoundError(f"Missing POTCAR files for elements: {', '.join(missing_elements)}")
    
    return ''.join(combined_content), selected_types

def get_ordered_elements(structure):
    """Get ordered list of elements in the structure"""
    elements = []
    for site in structure:
        element = str(site.specie)
        if element not in elements:
            elements.append(element)
    return elements

def prepare_vasp_inputs(structure, material_id, potcar_dir, base_dir="vasp_calculations", preferred_potcar_types=None, logger=None):
    """Prepare VASP input files"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Create material directory
    material_dir = os.path.join(base_dir, material_id)
    os.makedirs(material_dir, exist_ok=True)
    
    try:
        # Save POSCAR
        poscar = Poscar(structure)
        poscar_path = os.path.join(material_dir, "POSCAR")
        poscar.write_file(poscar_path)
        logger.info(f"Saved POSCAR to {poscar_path}")
        
        # Get element list and generate POTCAR
        ordered_elements = get_ordered_elements(structure)
        potcar_content, selected_types = combine_potcar_files(
            ordered_elements, 
            potcar_dir, 
            logger,
            preferred_potcar_types
        )
        
        # Save POTCAR
        potcar_path = os.path.join(material_dir, "POTCAR")
        with open(potcar_path, 'w') as f:
            f.write(potcar_content)
        logger.info(f"Saved combined POTCAR to {potcar_path}")
        
        # Save pseudopotential selection information
        potcar_info_path = os.path.join(material_dir, "POTCAR_INFO.json")
        potcar_info = {
            "elements": ordered_elements,
            "selected_potcar_types": selected_types
        }
        with open(potcar_info_path, 'w') as f:
            json.dump(potcar_info, f, indent=4)
        logger.info(f"Saved POTCAR information to {potcar_info_path}")
        
        return {
            "poscar_path": poscar_path,
            "potcar_path": potcar_path,
            "potcar_info_path": potcar_info_path,
            "elements": ordered_elements,
            "potcar_types": selected_types
        }
        
    except Exception as e:
        logger.error(f"Error preparing VASP inputs for {material_id}: {str(e)}")
        raise

def process_abo3_materials(api_key, potcar_dir, max_materials=5, crystal_system="cubic", preferred_potcar_types=None):
    """Main function to process ABO3 materials"""
    logger = setup_test_logging()
    logger.info("Starting ABO3 materials processing")
    logger.info(f"Parameters: max_materials={max_materials}, crystal_system={crystal_system}")
    logger.info(f"POTCAR directory: {potcar_dir}")
    
    try:
        with MPRester(api_key) as mpr:
            # Search for materials
            docs = mpr.summary.search(
                num_sites=5,
                elements=["O"],
                fields=["material_id", "formula_pretty", "structure"]
            )
            
            processed_materials = []
            for doc in docs:
                structure = doc.structure
                composition = structure.composition
                
                # Check if it is ABO3
                if composition["O"] == 3 and len(composition) == 3:
                    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
                    crystal_sys = analyzer.get_crystal_system().lower()
                    
                    if crystal_system is None or crystal_sys == crystal_system.lower():
                        try:
                            # Prepare VASP input files
                            vasp_inputs = prepare_vasp_inputs(
                                structure,
                                doc.material_id,
                                potcar_dir,
                                preferred_potcar_types=preferred_potcar_types,
                                logger=logger
                            )
                            
                            material_info = {
                                "material_id": doc.material_id,
                                "formula": doc.formula_pretty,
                                "crystal_system": crystal_sys,
                                "structure_info": vasp_inputs
                            }
                            
                            processed_materials.append(material_info)
                            logger.info(f"Successfully processed material: {doc.material_id}")
                            
                            if len(processed_materials) >= max_materials:
                                break
                                
                        except Exception as e:
                            logger.error(f"Error processing material {doc.material_id}: {str(e)}")
                            continue
            
            # Save processing results
            results_file = "processed_materials.json"
            with open(results_file, 'w') as f:
                json.dump(processed_materials, f, indent=4)
            logger.info(f"Saved processing results to {results_file}")
            
            return processed_materials
            
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

def main():
    """Main function"""
    # Configuration parameters
    api_key = ""  # Replace with your API key
    potcar_dir = ""  # Replace with your POTCAR directory path
    
    # Optional: Specify preferred pseudopotential types
    preferred_potcar_types = {
        "Fe": "Fe_pv",      # Prefer pv version for Fe
        "O": "O",           # Prefer standard version for O
        "Ti": "Ti_sv",      # Prefer sv version for Ti
        # Add more elements as needed
    }
    
    try:
        # Process materials
        materials = process_abo3_materials(
            api_key,
            potcar_dir,
            max_materials=5,
            crystal_system="cubic",
            preferred_potcar_types=preferred_potcar_types
        )
        
        # Print summary of results
        print("\nProcessing Summary:")
        print(f"Total materials processed: {len(materials)}")
        for material in materials:
            print(f"\nMaterial ID: {material['material_id']}")
            print(f"Formula: {material['formula']}")
            print(f"Selected POTCAR types: {material['structure_info']['potcar_types']}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()