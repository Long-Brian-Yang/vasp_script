import os
import json
import numpy as np
import logging
from datetime import datetime
import traceback
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def setup_logging(log_dir="logs"):
    """
    Set up logging configuration with timestamp
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vasp_workflow_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_abo3_materials(api_key, max_materials=100, crystal_system=None, symmetry_tolerance=0.1):
    """
    Get ABO3-type materials from Materials Project
    """
    logger = logging.getLogger(__name__)
    try:
        with MPRester(api_key) as mpr:
            criteria = {
                "nelements": 3,
                "elements": {"$all": ["O"]},
                "anonymous_formula": {"A": 1, "B": 1, "C": 3},
            }
            
            properties = ["material_id", "formula_pretty", "structure"]
            results = mpr.query(criteria=criteria, properties=properties)
            
            filtered_materials = []
            for result in results:
                composition = result["structure"].composition
                if composition["O"] == 3:
                    structure = result["structure"]
                    analyzer = SpacegroupAnalyzer(structure, symprec=symmetry_tolerance)
                    
                    crystal_sys = analyzer.get_crystal_system().lower()
                    spacegroup = analyzer.get_space_group_number()
                    
                    if crystal_system is None or crystal_sys == crystal_system.lower():
                        filtered_materials.append({
                            "material_id": result["material_id"],
                            "formula": result["formula_pretty"],
                            "crystal_system": crystal_sys,
                            "spacegroup": spacegroup,
                            "structure": structure
                        })
                        
                        if len(filtered_materials) >= max_materials:
                            break
            
            return filtered_materials
            
    except Exception as e:
        logger.error(f"Error in get_abo3_materials: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_ordered_elements(structure):
    """
    Get ordered list of elements from structure
    """
    elements = []
    for site in structure:
        element = str(site.specie)
        if element not in elements:
            elements.append(element)
    return elements

def combine_potcar_files(element_list, potcar_dir):
    """
    Combine POTCAR files for given elements
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Combining POTCAR files for elements: {element_list}")
    
    combined_content = []
    missing_elements = []
    
    for element in element_list:
        potcar_path = os.path.join(potcar_dir, f"{element}/POTCAR")
        
        if os.path.exists(potcar_path):
            logger.info(f"Found POTCAR for {element}")
            with open(potcar_path, 'r') as f:
                content = f.read()
                combined_content.append(content)
        else:
            logger.error(f"POTCAR not found for {element} at {potcar_path}")
            missing_elements.append(element)
    
    if missing_elements:
        raise FileNotFoundError(f"Missing POTCAR files for elements: {', '.join(missing_elements)}")
    
    return ''.join(combined_content)

def create_run_script(work_dir, nodes=1, ppn=24, queue="normal", walltime="72:00:00"):
    """
    Create a VASP run script
    """
    script_content = f"""#!/bin/bash
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -j oe

cd $PBS_O_WORKDIR

# Load required modules (modify according to your system)
module load intel/2020
module load mpi/intel/2020
module load vasp/5.4.4

# Run VASP
mpirun -np {nodes * ppn} vasp_std > vasp.out
"""
    
    script_path = os.path.join(work_dir, "run.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)

def prepare_calculation_directory(material, potcar_dir):
    """
    Prepare complete VASP calculation directory
    """
    logger = logging.getLogger(__name__)
    
    try:
        material_id = material["material_id"]
        structure = material["structure"]
        
        # Create work directory
        work_dir = f"vasp_calc_{material_id}"
        os.makedirs(work_dir, exist_ok=True)
        
        # Generate POSCAR
        poscar = Poscar(structure)
        poscar.write_file(os.path.join(work_dir, "POSCAR"))
        
        # Generate POTCAR
        ordered_elements = get_ordered_elements(structure)
        combined_potcar = combine_potcar_files(ordered_elements, potcar_dir)
        with open(os.path.join(work_dir, "POTCAR"), "w") as f:
            f.write(combined_potcar)
        
        # Generate KPOINTS
        kpoints = Kpoints.automatic_density(structure, 1000)
        kpoints.write_file(os.path.join(work_dir, "KPOINTS"))
        
        # Generate INCAR
        incar_dict = {
            "SYSTEM": f"ABO3_{material_id}",
            "PREC": "Accurate",
            "EDIFF": 1e-6,
            "ENCUT": 520,
            "IBRION": 2,
            "NSW": 99,
            "ISIF": 3,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LREAL": "Auto",
            "ALGO": "Fast",
            "LWAVE": ".FALSE.",
            "LCHARG": ".TRUE."
        }
        incar = Incar(incar_dict)
        incar.write_file(os.path.join(work_dir, "INCAR"))
        
        # Create run script
        create_run_script(work_dir)
        
        logger.info(f"Prepared calculation directory for {material_id}")
        return work_dir
        
    except Exception as e:
        logger.error(f"Error preparing calculation directory: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def parse_vasp_files(work_dir):
    """
    Parse VASP output files and save results
    """
    logger = logging.getLogger(__name__)
    try:
        poscar_path = os.path.join(work_dir, "POSCAR")
        outcar_path = os.path.join(work_dir, "OUTCAR")
        vasprun_path = os.path.join(work_dir, "vasprun.xml")
        output_json_path = os.path.join(work_dir, "dataset.json")
        
        # Check files existence
        for file_path in [poscar_path, outcar_path, vasprun_path]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        # Read structure
        poscar = Poscar.from_file(poscar_path)
        structure = poscar.structure
        lattice = structure.lattice.matrix.tolist()
        species = [str(sp) for sp in structure.species]
        coords = structure.frac_coords.tolist()

        # Initialize data structure
        data = {
            "structures": [{
                "lattice": lattice,
                "species": species,
                "coords": coords
            }],
            "labels": {
                "energies": [],
                "stresses": [],
                "forces": []
            }
        }

        # Extract forces from OUTCAR
        forces_extracted = False
        outcar = Outcar(outcar_path)
        if hasattr(outcar, 'ionic_steps') and outcar.ionic_steps:
            last_step_outcar = outcar.ionic_steps[-1]
            if 'forces' in last_step_outcar and last_step_outcar['forces'] is not None:
                forces = last_step_outcar['forces']
                if isinstance(forces, np.ndarray):
                    forces = forces.tolist()
                data["labels"]["forces"].append(forces)
                forces_extracted = True

        # Parse vasprun.xml
        vasprun = Vasprun(vasprun_path, parse_potcar_file=False, parse_dos=False, parse_eigen=False)
        energy = vasprun.final_energy
        data["labels"]["energies"].append(energy)

        ionic_steps = vasprun.ionic_steps
        if ionic_steps:
            last_step_vasprun = ionic_steps[-1]

            # Process stress tensor
            stress = last_step_vasprun["stress"]
            if isinstance(stress, list):
                stress_array = np.array(stress, dtype=float)
                stress_gpa = stress_array * 0.1  # Convert kBar to GPa
                stress_flat = stress_gpa.flatten().tolist()
                data["labels"]["stresses"].append(stress_flat)

            # Get forces if not already extracted
            if not forces_extracted:
                if "forces" in last_step_vasprun and last_step_vasprun["forces"] is not None:
                    forces = last_step_vasprun["forces"]
                    if isinstance(forces, np.ndarray):
                        forces = forces.tolist()
                    data["labels"]["forces"].append(forces)

        # Save results
        with open(output_json_path, "w") as f:
            json.dump(data, f, indent=4)
        
        return data
        
    except Exception as e:
        logger.error(f"Error parsing VASP files: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    # Set up logging
    logger = setup_logging()
    
    # Configuration parameters
    api_key = "YOUR_MP_API_KEY"  # Replace with your Materials Project API key
    potcar_dir = "/path/to/your/potcar/directory"  # Replace with your POTCAR path
    max_materials = 50
    crystal_system = "cubic"  # Options: 'cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'trigonal', 'monoclinic', 'triclinic'
    symmetry_tolerance = 0.1
    
    logger.info("Starting VASP workflow")
    logger.info(f"Configuration: max_materials={max_materials}, crystal_system={crystal_system}")
    
    try:
        # Get ABO3 materials
        logger.info(f"Retrieving {crystal_system if crystal_system else 'all'} crystal system ABO3 materials...")
        materials = get_abo3_materials(api_key, max_materials, crystal_system, symmetry_tolerance)
        logger.info(f"Found {len(materials)} matching ABO3 materials")
        
        # Process each material
        for material in materials:
            material_id = material["material_id"]
            formula = material["formula"]
            logger.info(f"\nProcessing material: {formula} ({material_id})")
            
            try:
                # Prepare calculation directory
                work_dir = prepare_calculation_directory(material, potcar_dir)
                logger.info(f"Prepared calculation directory: {work_dir}")
                
                # Optional: Submit job
                # Uncomment to submit job automatically
                # os.system(f"cd {work_dir} && qsub run.sh")
                
            except Exception as e:
                logger.error(f"Error processing material {material_id}: {str(e)}")
                continue
        
        logger.info("All materials processed successfully")
        
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("VASP workflow completed")

if __name__ == "__main__":
    main()