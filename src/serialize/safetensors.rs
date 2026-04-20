//! Safetensors format read/write for PyTorch/HuggingFace interop.

use std::collections::HashMap;
use std::path::Path;
use crate::tensor::tensor::Tensor;
use crate::tensor::dtype::DType;

/// Error type for safetensors operations.
#[derive(Debug, thiserror::Error)]
pub enum SafeTensorsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("format error: {0}")]
    Format(String),
    #[error("dtype not supported: {0:?}")]
    DtypeNotSupported(DType),
}

/// Save tensors to safetensors format.
pub fn save(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<(), SafeTensorsError> {
    let mut entries: Vec<(&str, &Tensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    entries.sort_by_key(|(k, _)| *k);

    // Build header JSON
    let mut header_data = Vec::new();
    let mut offset: usize = 0;
    for (name, tensor) in &entries {
        let dtype_str = dtype_to_safetensors(tensor.dtype())?;
        let shape = tensor.dims().to_vec();
        let entry = serde_json::json!({
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + tensor.numel() * tensor.dtype().size_of()]
        });
        header_data.push((*name, entry));
        offset += tensor.numel() * tensor.dtype().size_of();
    }

    let header_map: HashMap<&str, serde_json::Value> = header_data.iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    let header_json = serde_json::to_string(&header_map)
        .map_err(|e| SafeTensorsError::Format(e.to_string()))?;

    // Header format: 8-byte LE length prefix + JSON padded to 8-byte boundary + data
    let header_bytes = header_json.as_bytes();
    let padding = (8 - (8 + header_bytes.len()) % 8) % 8;
    let header_len = 8 + header_bytes.len() + padding;

    let mut file = std::fs::File::create(path)?;
    use std::io::Write;
    file.write_all(&(header_len as u64).to_le_bytes())?;
    file.write_all(header_bytes)?;
    file.write_all(&vec![b' '; padding])?;

    // Write tensor data
    for (_, tensor) in &entries {
        let bytes = tensor.storage().as_bytes();
        // For contiguous tensors, write from offset
        let start = tensor.offset * tensor.dtype().size_of();
        let end = start + tensor.numel() * tensor.dtype().size_of();
        file.write_all(&bytes[start..end])?;
    }

    Ok(())
}

/// Load tensors from safetensors format.
pub fn load(path: &Path) -> Result<HashMap<String, Tensor>, SafeTensorsError> {
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        return Err(SafeTensorsError::Format("file too small".into()));
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if data.len() < header_len {
        return Err(SafeTensorsError::Format("header extends past file".into()));
    }

    let header_json: HashMap<String, serde_json::Value> = serde_json::from_slice(&data[8..header_len])
        .map_err(|e| SafeTensorsError::Format(e.to_string()))?;

    let mut result = HashMap::new();
    for (name, meta) in &header_json {
        let dtype_str = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("F32");
        let shape: Vec<usize> = meta.get("shape")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
            .unwrap_or_default();
        let offsets = meta.get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect::<Vec<_>>())
            .unwrap_or_default();

        if offsets.len() != 2 { continue; }
        let (start, end) = (offsets[0], offsets[1]);
        let tensor_data = data[header_len + start..header_len + end].to_vec();
        let numel: usize = shape.iter().product::<usize>().max(1);
        let dtype = safetensors_to_dtype(dtype_str)?;

        let storage = crate::tensor::storage::Storage::from_bytes(tensor_data, dtype, numel);
        let tensor = Tensor::from_storage(storage, shape);
        result.insert(name.clone(), tensor);
    }

    Ok(result)
}

fn dtype_to_safetensors(dtype: DType) -> Result<&'static str, SafeTensorsError> {
    match dtype {
        DType::F16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::F32 => Ok("F32"),
        DType::F64 => Ok("F64"),
        _ => Err(SafeTensorsError::DtypeNotSupported(dtype)),
    }
}

fn safetensors_to_dtype(s: &str) -> Result<DType, SafeTensorsError> {
    match s {
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "F32" => Ok(DType::F32),
        "F64" => Ok(DType::F64),
        _ => Err(SafeTensorsError::Format(format!("unknown dtype: {}", s))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert("weight".into(), Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let path = std::env::temp_dir().join("flame_test.safetensors");
        save(&tensors, &path).unwrap();
        let loaded = load(&path).unwrap();
        assert!(loaded.contains_key("weight"));
        assert_eq!(loaded["weight"].dims(), &[2, 2]);
        let _ = std::fs::remove_file(&path);
    }
}
