# r3_tsm_r50_k400_diff_final

Deployment classifier for the exam abnormal-behavior prototype.

- Architecture: TSM-ResNet50
- Initialization: Kinetics-400 TSM
- ROI: R3 Compact Actor-Context
- Segments: 8
- Input: 224x224
- Classes: 5
- Development training samples: 509
- Fixed epochs: 8
- Epoch selection: median of 5-fold best epochs [10, 9, 7, 8, 5]

Important:
- model.pth is the deployment checkpoint.
- Use r3_roi_config.json to generate runtime ROI.
- Use deterministic validation/inference preprocessing, not training augmentation.
- This final development retrain does NOT create a new independent-test claim.

Production integration trong repository:

- Model/class order: `backend/app/ai/r3_model_loader.py`,
  `backend/app/ai/r3_predictor.py`.
- Validation/inference preprocessing: `backend/app/ai/r3_preprocessing.py`, đối
  chiếu `dataset_tsm_reference.py` trong package này.
- ROI geometry: `backend/app/ai/r3_roi.py` và `backend/app/ai/roi_processor.py`,
  lấy tham số từ `r3_roi_config.json` trong package này.
- Smoke/regression: `backend/tests/ai/test_r3_integration.py`.
- Audit, lệnh kiểm thử và giới hạn: `docs/R3_DEPLOYMENT_INTEGRATION.md`.

Package chưa có R3-compatible clip/golden ROI fixture. Không dùng ba golden
prediction E3 làm expected output R3. ROI vẫn do người dùng cấu hình cố định.
