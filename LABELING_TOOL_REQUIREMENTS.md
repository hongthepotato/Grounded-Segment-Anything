# Labeling Tool Requirements for COCO Format

## Core Principle: One Annotation = One Object = One Bbox

### ✅ Correct Approach

When user labels **multiple objects** in an image, create **separate annotations**:

```json
{
  "images": [
    {"id": 0, "width": 640, "height": 480, "file_name": "img1.jpg"}
  ],
  "categories": [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "car"}
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [[100, 100, 150, 100, 150, 200, 100, 200]],
      "bbox": null  // Backend will auto-generate
    },
    {
      "id": 1,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [[400, 300, 500, 300, 500, 450, 400, 450]],
      "bbox": null  // Backend will auto-generate
    }
  ]
}
```

**Result:** Backend generates two tight bboxes, one for each object ✅

---

### ❌ Incorrect Approach (Will Trigger Warning)

**DO NOT** put multiple disconnected objects in one annotation:

```json
{
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [
        [100, 100, 150, 100, 150, 200, 100, 200],     // Object 1
        [400, 300, 500, 300, 500, 450, 400, 450]      // Object 2 (far away!)
      ],
      "bbox": null
    }
  ]
}
```

**Result:** Backend generates ONE large bbox covering both objects + empty space ❌

```
Generated bbox: [100, 100, 400, 350]  // Covers huge empty space!
```

**Backend validation will warn:**
```
⚠️  Annotation 0 has suspicious bbox (bbox area >> mask area).
    This likely means multiple disconnected objects in one annotation.
    Please split these into separate annotations in your labeling tool.
```

---

## Frontend Implementation Checklist

### 1. One Mask Draw = One Annotation

When user draws a mask for an object:
- Generate unique annotation ID
- Store mask polygon coordinates
- Set `bbox: null` (backend auto-generates)
- Save as separate annotation entry

```javascript
// User draws mask for object 1
const annotation1 = {
  id: generateUniqueId(),
  image_id: currentImageId,
  category_id: selectedCategoryId,
  segmentation: [maskPolygonCoordinates],
  bbox: null,  // Backend handles this
  area: null   // Backend computes this
};

// User draws mask for object 2
const annotation2 = {
  id: generateUniqueId(),
  image_id: currentImageId,
  category_id: selectedCategoryId,
  segmentation: [maskPolygonCoordinates],
  bbox: null,
  area: null
};

annotations.push(annotation1, annotation2);
```

### 2. ID Requirements

#### Image IDs
- Must be non-negative integers (>= 0)
- Can be arbitrary (0, 1, 2, 5, 100, ...)
- Must be unique within the dataset

#### Annotation IDs
- Must be non-negative integers (>= 0)
- Can be arbitrary (0, 1, 2, 5, 100, ...)
- Must be unique within the dataset

#### Category IDs
- **CRITICAL:** Must satisfy `categories[i]['id'] == i`
- Categories must be ordered: 
  ```json
  [
    {"id": 0, "name": "person"},   // categories[0]
    {"id": 1, "name": "car"},      // categories[1]
    {"id": 2, "name": "dog"}       // categories[2]
  ]
  ```

### 3. Optional Fields (Backend Auto-Generates)

Your frontend can omit these fields, backend will compute them:

- `bbox`: Generated from segmentation mask
- `area`: Computed from segmentation mask or bbox
- `iscrowd`: Defaults to 0 if not provided

### 4. Polygon Format

**Frontend must use polygon format** (not RLE). Backend accepts both but frontend should only produce polygons.

#### Single polygon (simple objects):
```json
"segmentation": [
  [x1, y1, x2, y2, x3, y3, x4, y4]  // One polygon (at least 3 points = 6 coordinates)
]
```

#### Multiple polygons (complex objects):
For objects with multiple parts or holes (e.g., person with spread arms, donut):
```json
"segmentation": [
  [x1, y1, x2, y2, ...],  // Part 1: Main body
  [x5, y5, x6, y6, ...]   // Part 2: Left arm (connected to body)
]
```

**CRITICAL: Multiple polygons represent ONE object with multiple parts, NOT multiple separate objects!**

Examples:
- ✅ Person with spread arms → 1 annotation with 2-3 polygons (body, left arm, right arm)
- ✅ Donut → 1 annotation with 2 polygons (outer circle, inner hole)
- ❌ 2 people → Should be 2 annotations, each with 1+ polygons

If you have disconnected objects, create separate annotations!

#### Validation Rules (Backend will check):
- Each polygon must have **at least 6 coordinates** (3 points minimum for a triangle)
- Coordinates must be **even count** (x,y pairs)
- All coordinates must be **numbers** (int or float)
- No **empty polygons**: `[[]]` is invalid
- At least one polygon if segmentation is provided: `[]` is only OK for bbox-only annotations

---

## Validation Workflow

### Frontend → Backend Pipeline

```
1. User labels objects in frontend
   ↓
2. Frontend exports COCO JSON (with bbox=null)
   ↓
3. User runs: python cli/validate_dataset.py
   ↓
4. Backend validates format
   ↓
5. Backend auto-generates missing bbox/area
   ↓
6. Backend runs quality checks
   ↓
7. If suspicious bboxes detected → Warning (not error)
   ↓
8. Dataset ready for training
```

### Example Validation Output

```bash
$ python cli/validate_dataset.py --data-path annotations.json

✓ Format validation passed
✓ Auto-generated 150 bounding boxes from masks
✓ Auto-computed 150 areas

Quality Check Report:
  Total images: 100
  Total annotations: 150
  
  Warnings:
  ⚠️  5 annotations (3.3%) have suspicious bboxes (bbox area >> mask area).
      This likely means multiple disconnected objects in one annotation.
      COCO format requires: one object = one annotation.
      Please split these into separate annotations in your labeling tool.
      
      Affected annotations: [12, 45, 67, 89, 123]
```

---

## Design Rationale

### Why Not Auto-Split Disconnected Masks?

**Option A: Auto-split into multiple annotations** ❌
```python
# Input: 1 annotation with 2 disconnected masks
# Output: 2 annotations with 2 bboxes
```

**Problems:**
1. How to assign new annotation IDs? (could conflict with existing IDs)
2. How to determine if masks are "truly disconnected" vs "parts of one object"?
3. Adds complexity to preprocessing (should be simple)
4. Hides data quality issues instead of exposing them

**Option B: Warn user to fix in labeling tool** ✅
```python
# Backend detects suspicious data → warns user → user fixes at source
```

**Benefits:**
1. Data quality issues visible immediately
2. Frontend enforces correct structure from the start
3. Backend stays simple (one annotation in → one bbox out)
4. Users learn correct labeling practices

### Why One Bbox Per Annotation?

This is the **COCO format standard**:
- One annotation = One semantic object
- One object = One bounding box
- Multiple objects = Multiple annotations

Benefits:
- Simple data structure (no nested arrays)
- Easy to index: `annotations[i]['bbox']` is always a single bbox
- Matches object detection model outputs (one prediction per object)

---

## Summary for Frontend Developer

**Your responsibility:**
- Create **one annotation per object**
- Store segmentation masks (polygon format)
- Set `bbox: null` and `area: null` (backend handles these)
- Ensure category IDs follow the ordering requirement

**Backend's responsibility:**
- Generate bboxes from masks
- Compute areas
- Validate data format
- Warn about quality issues

**If you get a "suspicious bbox" warning:**
- Review those specific annotations in your labeling tool
- Check if one annotation contains multiple disconnected objects
- Split them into separate annotations
- Re-export and validate again

---

## Questions for Frontend Developer

1. **How do you handle multi-object labeling?**
   - Does each mask draw create a new annotation? (should be YES)
   - Or does user add multiple masks to one annotation? (should be NO)

2. **How do you assign category IDs?**
   - Need to ensure categories[0].id=0, categories[1].id=1, etc.

3. **What polygon format do you export?**
   - Should be: `[x1, y1, x2, y2, x3, y3, ...]` (flat array)

4. **Do you generate bbox/area in frontend?**
   - If yes, we can validate them
   - If no, backend will auto-generate (recommended)

