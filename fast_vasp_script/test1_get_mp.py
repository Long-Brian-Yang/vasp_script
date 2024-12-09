import os
import json
import logging
from datetime import datetime
from pymatgen.ext.matproj import MPRester
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

def save_structure_files(structure, material_id, base_dir="structures"):
    """Save structure files in different formats"""
    # Create directory for saving files
    material_dir = os.path.join(base_dir, material_id)
    os.makedirs(material_dir, exist_ok=True)
    
    # Save as CIF format
    cif_writer = CifWriter(structure)
    cif_path = os.path.join(material_dir, f"{material_id}.cif")
    cif_writer.write_file(cif_path)
    
    # Save as POSCAR format
    poscar = Poscar(structure)
    poscar_path = os.path.join(material_dir, "POSCAR")
    poscar.write_file(poscar_path)
    
    return {
        "cif_path": cif_path,
        "poscar_path": poscar_path
    }

def get_abo3_materials(api_key, max_materials=5, crystal_system="cubic", symmetry_tolerance=0.1):
    """Get ABO3 materials and save structure files"""
    logger = setup_test_logging()
    logger.info(f"Starting ABO3 materials search with parameters:")
    logger.info(f"- max_materials: {max_materials}")
    logger.info(f"- crystal_system: {crystal_system}")
    logger.info(f"- symmetry_tolerance: {symmetry_tolerance}")
    
    try:
        with MPRester(api_key) as mpr:
            # Use the new API query method
            docs = mpr.summary.search(
                num_sites=5,  # Total of 5 atoms in ABO3
                elements=["O"],  # Must contain oxygen
                fields=["material_id", "formula_pretty", "structure"]
            )
            
            # Save results
            materials_data = []
            
            for doc in docs:
                structure = doc.structure
                composition = structure.composition
                
                # Check if it matches the ABO3 ratio
                if composition["O"] == 3 and len(composition) == 3:
                    analyzer = SpacegroupAnalyzer(structure, symprec=symmetry_tolerance)
                    crystal_sys = analyzer.get_crystal_system().lower()
                    spacegroup = analyzer.get_space_group_number()
                    
                    # Check crystal system
                    if crystal_system is None or crystal_sys == crystal_system.lower():
                        # Save structure files
                        structure_files = save_structure_files(structure, doc.material_id)
                        
                        material_info = {
                            "material_id": doc.material_id,
                            "formula": doc.formula_pretty,
                            "crystal_system": crystal_sys,
                            "spacegroup": spacegroup,
                            "composition": dict(composition.get_el_amt_dict()),
                            "structure_files": structure_files
                        }
                        materials_data.append(material_info)
                        logger.info(f"Processed material: {doc.material_id}")
                        logger.info(f"- Formula: {doc.formula_pretty}")
                        logger.info(f"- Files saved: {structure_files}")
                        
                        if len(materials_data) >= max_materials:
                            break
            
            # Save material information to JSON file
            with open("abo3_materials.json", "w") as f:
                json.dump(materials_data, f, indent=4)
            
            logger.info(f"\nTotal materials processed: {len(materials_data)}")
            return materials_data
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

def main():
    """Main function"""
    api_key = ""
    
    # Get and save material data
    materials = get_abo3_materials(
        api_key,
        max_materials=5,
        crystal_system="cubic",
        symmetry_tolerance=0.1
    )
    
    # Print retrieved material information
    # print("\nRetrieved Materials:")
    # for material in materials:
    #     print(f"\nMaterial ID: {material['material_id']}")
    #     print(f"Formula: {material['formula']}")
    #     print(f"Crystal System: {material['crystal_system']}")
    #     print(f"Structure files:")
    #     print(f"- CIF: {material['structure_files']['cif_path']}")
    #     print(f"- POSCAR: {material['structure_files']['poscar_path']}")

if __name__ == "__main__":
    main()