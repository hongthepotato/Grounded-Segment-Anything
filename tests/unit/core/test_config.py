"""
Unit tests for core.config module.

Tests configuration loading, saving, merging, and generation functionality.
Uses strict testing approach: we control exact input, test only target function.
"""

import pytest
from core.config import (
    load_config,
    save_config,
    load_json,
    save_json,
    merge_configs
)


@pytest.mark.unit
class TestLoadConfig:
    """Test YAML configuration loading."""

    def test_load_config_success(self, temp_dir):
        """Test loading valid YAML config file."""
        # Create raw YAML content - we control exact input
        yaml_content = """
learning_rate: 0.0001
batch_size: 8
epochs: 50
optimizer: AdamW
weight_decay: 0.0001
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:
    - value_proj
    - output_proj
dataset:
  num_classes: 3
  class_names:
    - background
    - cat
    - dog
"""
        config_path = temp_dir / 'test_config.yaml'
        config_path.write_text(yaml_content)

        # Test ONLY load_config
        loaded = load_config(str(config_path))

        # Verify it parsed correctly
        assert loaded['learning_rate'] == 0.0001
        assert loaded['batch_size'] == 8
        assert loaded['epochs'] == 50
        assert loaded['lora']['r'] == 16
        assert loaded['lora']['lora_alpha'] == 32
        assert loaded['dataset']['num_classes'] == 3
        assert loaded['dataset']['class_names'] == ['background', 'cat', 'dog']

    def test_load_config_with_nested_structure(self, temp_dir):
        """Test loading deeply nested YAML structure."""
        yaml_content = """
model:
  backbone:
    type: resnet
    layers: 50
    pretrained: true
  head:
    type: classification
    num_classes: 10
"""
        config_path = temp_dir / 'nested.yaml'
        config_path.write_text(yaml_content)

        loaded = load_config(str(config_path))

        assert loaded['model']['backbone']['type'] == 'resnet'
        assert loaded['model']['backbone']['layers'] == 50
        assert loaded['model']['backbone']['pretrained'] is True
        assert loaded['model']['head']['num_classes'] == 10

    def test_load_config_file_not_found(self, temp_dir):
        """Test loading non-existent config raises FileNotFoundError."""
        fake_path = temp_dir / 'nonexistent.yaml'

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(fake_path))


@pytest.mark.unit
class TestSaveConfig:
    """Test YAML configuration saving."""

    def test_save_config_creates_file(self, temp_dir):
        """Test saving config creates proper YAML file."""
        config = {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'lora': {'r': 16, 'alpha': 32}
        }
        output_path = temp_dir / 'output_config.yaml'

        save_config(config, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert 'learning_rate' in content
        assert 'batch_size' in content
        assert 'lora:' in content

    def test_save_config_creates_parent_dirs(self, temp_dir):
        """Test save_config creates parent directories if they don't exist."""
        config = {'test': 'value'}
        nested_path = temp_dir / 'level1' / 'level2' / 'config.yaml'

        save_config(config, str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_config_overwrites_existing(self, temp_dir):
        """Test that save_config overwrites existing files."""
        config_path = temp_dir / 'config.yaml'

        config_path.write_text("old: data\n")

        new_config = {'new': 'data'}
        save_config(new_config, str(config_path))

        content = config_path.read_text()
        assert 'old' not in content
        assert 'new' in content


@pytest.mark.unit
class TestLoadJson:
    """Test JSON file loading."""

    def test_load_json_success(self, temp_dir):
        """Test loading valid JSON file."""
        json_content = '{"key": "value", "number": 42, "nested": {"item": "test"}}'
        json_path = temp_dir / 'test.json'
        json_path.write_text(json_content)

        loaded = load_json(str(json_path))

        assert loaded['key'] == 'value'
        assert loaded['number'] == 42
        assert loaded['nested']['item'] == 'test'

    def test_load_json_with_arrays(self, temp_dir):
        """Test loading JSON with arrays."""
        json_content = '{"items": [1, 2, 3], "names": ["a", "b", "c"]}'
        json_path = temp_dir / 'array.json'
        json_path.write_text(json_content)

        loaded = load_json(str(json_path))

        assert loaded['items'] == [1, 2, 3]
        assert loaded['names'] == ['a', 'b', 'c']

    def test_load_json_file_not_found(self, temp_dir):
        """Test loading non-existent JSON raises FileNotFoundError."""
        fake_path = temp_dir / 'nonexistent.json'

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            load_json(str(fake_path))


@pytest.mark.unit
class TestSaveJson:
    """Test JSON file saving."""

    def test_save_json_creates_file(self, temp_dir):
        """Test saving JSON creates proper file."""
        data = {'test': 'data', 'nested': {'value': 123}}
        output_path = temp_dir / 'output.json'

        save_json(data, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert '"test"' in content
        assert '"nested"' in content

    def test_save_json_creates_parent_dirs(self, temp_dir):
        """Test save_json creates parent directories."""
        data = {'test': 'value'}
        nested_path = temp_dir / 'dir1' / 'dir2' / 'data.json'

        save_json(data, str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()


@pytest.mark.unit
class TestMergeConfigs:
    """Test configuration merging logic."""

    def test_merge_flat_configs(self):
        """Test merging flat (non-nested) configs."""
        base = {'a': 1, 'b': 2, 'c': 3}
        override = {'b': 20, 'd': 4}
        result = merge_configs(base, override)
        assert result == {'a': 1, 'b': 20, 'c': 3, 'd': 4}

    def test_merge_nested_configs(self):
        """Test merging nested configs recursively."""
        base = {
            'a': 1,
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.1
            }
        }
        override = {
            'lora': {
                'r': 32,  # Override
                'beta': 64  # New key
            },
            'b': 2  # New top-level key
        }

        result = merge_configs(base, override)

        assert result['a'] == 1
        assert result['b'] == 2
        assert result['lora']['r'] == 32
        assert result['lora']['alpha'] == 32
        assert result['lora']['dropout'] == 0.1
        assert result['lora']['beta'] == 64

    def test_merge_does_not_modify_originals(self):
        """Test that merge_configs doesn't modify input dicts."""
        base = {'a': 1, 'b': {'c': 2}}
        override = {'b': {'d': 3}}

        base_copy = base.copy()
        override_copy = override.copy()
        merge_configs(base, override)

        assert base == base_copy
        assert override == override_copy

    def test_merge_configs_empty_override(self):
        """Test merging with empty override dict."""
        base = {'a': 1, 'b': 2}
        override = {}
        result = merge_configs(base, override)

        assert result == base

    def test_merge_configs_empty_base(self):
        """Test merging with empty base dict."""
        base = {}
        override = {'a': 1, 'b': 2}
        result = merge_configs(base, override)

        assert result == override
