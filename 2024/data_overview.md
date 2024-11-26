# Cereal Manufacturing Process Documentation

## Process Overview

```mermaid
flowchart TB
    subgraph Cooking
        cook[Cooking Process]
        ct[Cook Temp]
        ctime[Cook Time]
        cs[Cooker Speed]
        cook --> qc1[Quality Check 1]
    end
    
    subgraph Drying1[Drying Phase 1]
        dry1[Dryer 1]
        dt1[Drying Temp]
        dtime1[Drying Time]
        ds1[Drying Speed]
        dry1 --> qc1
    end
    
    subgraph QC1[Quality Check 1]
        qc1 --> moisture1[Moisture]
        qc1 --> color1[Color]
    end
    
    subgraph Flaking
        flake[Flaking Process]
        fg[Rolls Gap]
        fs[Rolls Speed]
        flake --> qc2[Quality Check 2]
    end
    
    subgraph QC2[Quality Check 2]
        qc2 --> moisture2[Moisture 2]
        qc2 --> thickness[Thickness]
        qc2 --> diameter[Diameter]
    end
    
    subgraph Coating
        coat[Coating Process]
        cf[Coating Flow]
        cs2[Conveyor Speed]
        coat --> qc3[Quality Check 3]
    end
    
    subgraph Drying2[Drying Phase 2]
        dry2[Dryer 2]
        dt2[Drying Temp]
        dtime2[Drying Time]
        ds2[Drying Speed]
        dry2 --> qc3
    end
    
    subgraph QC3[Quality Check 3]
        qc3 --> moisture3[Final Moisture]
        qc3 --> sugar[Sugar]
        qc3 --> vitamins[Vitamins]
        qc3 --> color2[Color Final]
    end
    
    Cooking --> Drying1
    Drying1 --> Flaking
    Flaking --> Coating
    Coating --> Drying2
```

## Process Parameters and Specifications

### 1. Cooking Process
- **Cook Temperature**
  - Range: 450°F ± 10°F
  - Relations: High correlation with Color
- **Cook Time**
  - Range: 20-30 minutes
- **Cooker Speed**
  - Range: 50 rpm ± 5 rpm

### 2. Dryer 1
- **Drying Temperature**
  - Range: 250°F ± 10°F
  - Relations: High correlation with Moisture
- **Drying Time**
  - Range: 10 minutes ± 1 minute
- **Drying Speed**
  - Range: 400 ft/min ± 20 ft/min

### 3. Quality Checks 1
- **Moisture**
  - Range: 0-100%
  - Target: 20% ± 2%
  - Critical: High relation to flaking quality
- **Color**
  - Range: 1-10
  - Target: 6-8
  - Note: Marketing requirement, no downstream impact

### 4. Flaking Process
- **Flaking Rolls Gap**
  - Range: 2mm ± 0.1mm
  - Critical: High relation to Thickness and Diameter
- **Flaking Rolls Speed**
  - Range: 300 rpm ± 10 rpm

### 5. Quality Checks 2
- **Moisture 2**
  - Range: 0-100%
  - Target: 20% ± 2%
- **Thickness**
  - Range: 0-1mm
  - Target: 0.5-0.9mm
  - Critical: Direct impact on First-Pass Quality
- **Diameter**
  - Range: 2-10mm
  - Target: 4-8mm
  - Critical: Direct impact on First-Pass Quality

### 6. Coating Process
- **Coating Flow**
  - Range: 2 gal/min ± 0.1 gal/min
  - Relations: High correlation with Sugar and Vitamins content
- **Conveyor Speed**
  - Range: 400 ft/min ± 10 ft/min

### 7. Dryer 2
- **Drying Temperature**
  - Range: 250°F ± 10°F
  - Relations: High correlation with Final Moisture
- **Drying Time**
  - Range: 10 minutes ± 1 minute
- **Drying Speed**
  - Range: 400 ft/min ± 20 ft/min

### 8. Quality Checks 3
- **Final Moisture**
  - Range: 0-100%
  - Target: 20% ± 2%
  - Critical: Out of spec requires scrapping
- **Sugar**
  - Range: 0-100%
  - Target: 40% ± 5%
  - Critical: Out of spec requires scrapping
- **Vitamins**
  - Range: 0-100%
  - Target: 10% ± 2%
  - Critical: Out of spec requires scrapping
- **Color Final**
  - Range: 1-10
  - Target: 6-8
  - Note: Marketing requirement

## Key Performance Indicators (KPIs)

```mermaid
flowchart LR
    subgraph KPIs
        direction TB
        throughput[Throughput\nlbs/hr]
        quality[First-Pass Quality\n%]
        yield[First-Pass Yield\n%]
        scrap[Scrap\nlbs]
    end
    
    moisture1 & color1 --> throughput
    thickness & diameter --> quality
    quality --> yield
    yield & throughput --> scrap
```

### Critical Relationships
1. **First-Pass Quality**
   - Primarily determined by Thickness and Diameter specifications
   - Must meet all quality parameters to achieve 100%

2. **First-Pass Yield**
   - Directly influenced by First-Pass Quality
   - Represents actual production efficiency

3. **Scrap**
   - Generated from:
     - Out-of-spec thickness/diameter
     - Moisture content deviation
     - Sugar/Vitamin content deviation
   - Inversely related to First-Pass Yield

4. **Throughput**
   - Influenced by:
     - Cooker Speed
     - Drying Speeds
     - Moisture Content