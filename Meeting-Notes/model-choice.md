# Comparing DiffTraj, Diff-RNTraj, and HOSER for Taxi GPS Data Generation

---

## 1. **Input Data Format**

| Aspect                | DiffTraj                        | Diff-RNTraj                                 | HOSER                                         |
|-----------------------|---------------------------------|---------------------------------------------|-----------------------------------------------|
| Trajectory Structure  | Sequence of (lat, lon) GPS pts  | Sequence of road segment IDs (+ GPS coords) | Sequence of (lat, lon) GPS points             |
| Context Features      | Per-trip scalar features         | Road, POI, grid, rates, segment features    | Trip-level or per-point features supported    |
| File Format           | `.npy`, `.csv`                  | `.txt`, `.csv`, custom batch formats        | `.csv`, `.txt`, `.pkl`, numpy arrays          |
| Loader Output         | `[batch, 2, traj_length]`, `[batch, 8]` | Multiple tensors (IDs, rates, POI, etc.)   | `[batch, traj_length, 2]` (+ timestamps/speed)|

**For taxi data like `id, timestamp, lon, lat, angle, speed`, HOSER directly supports this format.**
- HOSER is designed for raw GPS trajectories with rich per-point metadata, requiring minimal conversion.

---

## 2. **Preprocessing Pipeline**

| Step                      | DiffTraj                                    | Diff-RNTraj                        | HOSER                                    |
|---------------------------|---------------------------------------------|------------------------------------|-------------------------------------------|
| Resampling                | Utility provided, but not integrated        | Handled as part of map-matching    | Built-in or easily added (interpolation)  |
| Normalization             | Expected, not implemented                   | Explicitly performed (min-max, log)| Built-in or easily added                  |
| Map-matching              | Not supported                               | Fully supported                    | Not needed for raw GPS (unless desired)   |
| Feature Extraction        | Not implemented, user responsibility        | Built-in scripts/utilities         | Supports timestamps, speed, etc.          |
| Data Organization         | User must save `.npy` files manually        | Automated batch creation           | Directly loads tabular GPS data           |

**HOSER is the least intrusive for your data—you can use your taxi GPS records almost as-is without extensive preprocessing.**

---

## 3. **Output Format**

| Aspect           | DiffTraj                             | Diff-RNTraj                              | HOSER                                  |
|------------------|-------------------------------------|-------------------------------------------|----------------------------------------|
| Output Type      | Sequence of GPS points               | Sequence of road segment IDs (+ GPS/rates)| Sequence of GPS points                 |
| Visualization    | Direct GPS plotting                  | Requires mapping IDs to network           | Direct GPS plotting                    |
| File Format      | `.npy`, `.csv`                       | `.txt`, `.csv` (with various attributes)  | `.csv`, `.txt`, numpy arrays           |

**HOSER will output generated trajectories in the same format as your input (GPS points with optional metadata), compatible with mapping and analysis tools.**

---

## 4. **When is HOSER the Better Choice?**

### **Advantages for Taxi GPS Data**
- **Minimal Data Alteration:**  
  - Directly ingests per-point GPS data in standard taxi fleet formats.
  - No need to convert to road segment IDs or engineer additional network features.
- **Rich Metadata Support:**  
  - Can use timestamps, speed, angle, and taxi IDs for more nuanced modeling.
- **Simple Preprocessing:**  
  - Only basic cleaning/interpolation required; your existing data structure fits.
- **Output Consistency:**  
  - Generated data will match your input structure, making downstream analysis seamless.

### **Limitations of DiffTraj/Diff-RNTraj for Your Case**
- **DiffTraj:**  
  - Requires manual preprocessing and data shaping.
  - Context features are limited.
- **Diff-RNTraj:**  
  - Requires converting GPS points to road segment sequences.
  - Preprocessing pipeline is geared toward network-constrained data, which may not align with your taxi fleet’s raw GPS format.

---

## 5. **Recommendation**

**Choose HOSER if:**
- You have raw taxi GPS data (`id, timestamp, lon, lat, angle, speed`).
- You want to avoid altering your data format or structure.
- You need fast, direct integration and output that matches your input.

**Choose DiffTraj or Diff-RNTraj only if:**
- You require map-matched, network-aware trajectories, or need to generate data with explicit road network constraints.

---

## 6. **References**

- [DiffTraj Repository](https://github.com/Yasoz/DiffTraj)
- [Diff-RNTraj Repository](https://github.com/wtl52656/Diff-RNTraj)
- [HOSER Repository](https://github.com/caoji2001/HOSER)

---

**In summary:**  
**HOSER** is the best choice for generating synthetic taxi GPS trajectories when you want to preserve your existing data structure and avoid complex preprocessing or format conversion.