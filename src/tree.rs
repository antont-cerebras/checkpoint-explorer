use std::collections::HashMap;

/// How a tensor is stored on disk, for formats (like HDF5) that may compress.
// `Raw`/`Compressed` are only constructed by the HDF5 reader; without that
// feature they are still matched in the UI but never built.
#[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
#[derive(Debug, Clone)]
pub enum Storage {
    /// Compression is not tracked for this format (e.g. safetensors / GGUF).
    Unknown,
    /// Stored uncompressed on disk.
    Raw,
    /// Compressed on disk with `codec`; `stored_bytes` is the on-disk size.
    Compressed { codec: String, stored_bytes: usize },
}

/// Where a tensor's data sits within its file, by format.
// `Chunked` is only constructed by the HDF5 reader.
#[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
#[derive(Debug, Clone)]
pub enum Layout {
    /// Layout not tracked.
    None,
    /// safetensors: `[start, end)` byte range within the file's data blob.
    ByteRange { start: u64, end: u64 },
    /// GGUF: byte offset of the tensor's data within the tensor-data region.
    Offset(u64),
    /// HDF5: chunked storage with the given chunk shape and chunk count.
    Chunked {
        chunk: Vec<usize>,
        num_chunks: usize,
    },
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Logical (uncompressed) size in bytes: `num_elements * dtype_size`.
    pub size_bytes: usize,
    pub num_elements: usize,
    /// On-disk storage / compression status.
    pub storage: Storage,
    /// Absolute path of the file this tensor was loaded from.
    pub source_path: String,
    /// Where the tensor's data lives within its file.
    pub layout: Layout,
}

impl TensorInfo {
    /// The size actually occupied on disk: the compressed size when stored
    /// compressed, otherwise the logical size.
    pub fn on_disk_size(&self) -> usize {
        match &self.storage {
            Storage::Compressed { stored_bytes, .. } => *stored_bytes,
            _ => self.size_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetadataInfo {
    pub name: String,
    pub value: String,
    pub value_type: String,
}

#[derive(Debug, Clone)]
pub enum TreeNode {
    Group {
        name: String,
        children: Vec<TreeNode>,
        expanded: bool,
        tensor_count: usize,
        /// Total number of parameters (elements) across descendant tensors.
        params: usize,
        total_size: usize,
        /// Sum of descendant tensors' on-disk sizes (compressed where
        /// applicable). Equals `total_size` when nothing is compressed.
        stored_size: usize,
    },
    Tensor {
        info: TensorInfo,
        /// Compacted display label: when a chain of single-child groups collapses
        /// down to this lone tensor, the merged path (e.g. `self_attn.k_norm.weight`)
        /// is shown on one row. `None` renders the plain last segment of `name`.
        label: Option<String>,
    },
    Metadata {
        info: MetadataInfo,
    },
}

impl TreeNode {
    pub fn name(&self) -> &str {
        match self {
            TreeNode::Group { name, .. } => name,
            TreeNode::Tensor { info, .. } => &info.name,
            TreeNode::Metadata { info } => &info.name,
        }
    }
}

/// The last path segment of a tensor name, treating `.` and `__` as separators
/// (so `…_down_proj_weight__variant` yields `variant`). Used for the leaf label.
pub fn last_segment(name: &str) -> &str {
    let after = name.rsplit("__").next().unwrap_or(name);
    after.rsplit('.').next().unwrap_or(after)
}

pub fn natural_sort_key(name: &str) -> Vec<NaturalSortItem> {
    let mut result = Vec::new();
    let mut current_number = String::new();
    let mut current_text = String::new();

    for ch in name.chars() {
        if ch.is_ascii_digit() {
            if !current_text.is_empty() {
                result.push(NaturalSortItem::Text(current_text.clone()));
                current_text.clear();
            }
            current_number.push(ch);
        } else {
            if !current_number.is_empty() {
                if let Ok(num) = current_number.parse::<u32>() {
                    result.push(NaturalSortItem::Number(num));
                } else {
                    result.push(NaturalSortItem::Text(current_number.clone()));
                }
                current_number.clear();
            }
            current_text.push(ch);
        }
    }

    if !current_number.is_empty() {
        if let Ok(num) = current_number.parse::<u32>() {
            result.push(NaturalSortItem::Number(num));
        } else {
            result.push(NaturalSortItem::Text(current_number));
        }
    }
    if !current_text.is_empty() {
        result.push(NaturalSortItem::Text(current_text));
    }

    result
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NaturalSortItem {
    Text(String),
    Number(u32),
}

pub struct TreeBuilder;

impl TreeBuilder {
    pub fn build_tree_mixed(tensors: &[TensorInfo], metadata: &[MetadataInfo]) -> Vec<TreeNode> {
        let mut tree = Vec::new();

        // Add metadata as a separate group
        if !metadata.is_empty() {
            let mut metadata_children = Vec::new();
            for meta in metadata {
                metadata_children.push(TreeNode::Metadata { info: meta.clone() });
            }
            metadata_children.sort_by_key(|a| natural_sort_key(a.name()));

            tree.push(TreeNode::Group {
                name: "🔧 Metadata".to_string(),
                children: metadata_children,
                expanded: false,
                tensor_count: 0,
                params: 0,
                total_size: 0,
                stored_size: 0,
            });
        }

        // Build tensor tree
        let tensor_tree = Self::build_tree(tensors);
        tree.extend(tensor_tree);

        tree
    }

    pub fn build_tree(tensors: &[TensorInfo]) -> Vec<TreeNode> {
        let mut root_map: HashMap<String, Vec<TensorInfo>> = HashMap::new();

        for tensor in tensors {
            // Group on `.` and `__`: dotted names (HDF5 / safetensors) fold as
            // before, while underscore-flattened `.npy` names like
            // `…_down_proj_weight__variant` fold on the `__` boundary. Mapping
            // `__` → `.` keeps the rest of the path logic uniform; single `_`
            // (within a module name) is left untouched.
            let path = tensor.name.replace("__", ".");
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() > 1 {
                let prefix = parts[0].to_string();
                root_map.entry(prefix).or_default().push(tensor.clone());
            } else {
                root_map
                    .entry("_root".to_string())
                    .or_default()
                    .push(tensor.clone());
            }
        }

        let mut tree = Vec::new();
        for (prefix, mut tensors) in root_map {
            if prefix == "_root" {
                for tensor in tensors {
                    tree.push(TreeNode::Tensor {
                        info: tensor,
                        label: None,
                    });
                }
            } else {
                tensors.sort_by_key(|a| natural_sort_key(&a.name));
                let tensor_count = tensors.len();
                let params = tensors.iter().map(|t| t.num_elements).sum();
                let total_size = tensors.iter().map(|t| t.size_bytes).sum();
                let stored_size = tensors.iter().map(|t| t.on_disk_size()).sum();

                let children = Self::build_subtree(&tensors, &prefix);

                tree.push(TreeNode::Group {
                    name: prefix,
                    children,
                    expanded: true,
                    tensor_count,
                    params,
                    total_size,
                    stored_size,
                });
            }
        }

        tree.sort_by_key(|a| natural_sort_key(a.name()));
        // IDE-style "compact folders": collapse chains of single-child groups,
        // so a lone deeply-nested tensor shows as one `a.b.c.weight` row.
        compact_nodes(tree)
    }

    fn build_subtree(tensors: &[TensorInfo], prefix: &str) -> Vec<TreeNode> {
        let mut groups: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        let mut direct_tensors = Vec::new();

        for tensor in tensors {
            // Same `__` → `.` normalisation as `build_tree`, so the recursive
            // prefix-stripping treats both separators uniformly.
            let path = tensor.name.replace("__", ".");
            let remaining = path.strip_prefix(&format!("{prefix}.")).unwrap_or(&path);
            let parts: Vec<&str> = remaining.split('.').collect();

            if parts.len() == 1 {
                direct_tensors.push(tensor.clone());
            } else {
                let next_prefix = parts[0].to_string();
                groups.entry(next_prefix).or_default().push(tensor.clone());
            }
        }

        let mut result = Vec::new();

        for tensor in direct_tensors {
            result.push(TreeNode::Tensor {
                info: tensor,
                label: None,
            });
        }

        for (group_name, group_tensors) in groups {
            let tensor_count = group_tensors.len();
            let params = group_tensors.iter().map(|t| t.num_elements).sum();
            let total_size = group_tensors.iter().map(|t| t.size_bytes).sum();
            let stored_size = group_tensors.iter().map(|t| t.on_disk_size()).sum();
            let full_prefix = format!("{prefix}.{group_name}");
            let children = Self::build_subtree(&group_tensors, &full_prefix);

            result.push(TreeNode::Group {
                name: group_name,
                children,
                expanded: false,
                tensor_count,
                params,
                total_size,
                stored_size,
            });
        }

        result.sort_by_key(|a| natural_sort_key(a.name()));
        result
    }

    /// Expand every group on the path to the tensor named `name`, so it becomes
    /// visible (selectable) in the flattened tree. Returns whether it was found.
    pub fn expand_to_tensor(nodes: &mut [TreeNode], name: &str) -> bool {
        for node in nodes.iter_mut() {
            if let TreeNode::Tensor { info, .. } = node
                && info.name == name
            {
                return true;
            } else if let TreeNode::Group {
                children, expanded, ..
            } = node
                && Self::expand_to_tensor(children, name)
            {
                *expanded = true;
                return true;
            }
        }
        false
    }

    /// Recursively set the `expanded` flag on every group in the tree.
    pub fn set_all_expanded(nodes: &mut [TreeNode], expanded: bool) {
        for node in nodes {
            if let TreeNode::Group {
                expanded: node_expanded,
                children,
                ..
            } = node
            {
                *node_expanded = expanded;
                Self::set_all_expanded(children, expanded);
            }
        }
    }

    pub fn flatten_tree(tree: &[TreeNode]) -> Vec<(TreeNode, usize)> {
        let mut flattened = Vec::new();
        for node in tree {
            Self::flatten_node(node, 0, &mut flattened);
        }
        flattened
    }

    fn flatten_node(node: &TreeNode, depth: usize, flattened: &mut Vec<(TreeNode, usize)>) {
        flattened.push((node.clone(), depth));

        if let TreeNode::Group {
            children, expanded, ..
        } = node
            && *expanded
        {
            for child in children {
                Self::flatten_node(child, depth + 1, flattened);
            }
        }
    }

    pub fn toggle_node_by_index(target_idx: usize, nodes: &mut [TreeNode]) -> bool {
        let mut current_idx = 0;
        Self::toggle_node_by_index_recursive(target_idx, nodes, &mut current_idx)
    }

    fn toggle_node_by_index_recursive(
        target_idx: usize,
        nodes: &mut [TreeNode],
        current_idx: &mut usize,
    ) -> bool {
        for node in nodes {
            // Check if this is the target node
            if *current_idx == target_idx {
                if let TreeNode::Group { expanded, .. } = node {
                    *expanded = !*expanded;
                    return true;
                }
                return false; // Target was not a group
            }

            // Increment for this node
            *current_idx += 1;

            // If it's an expanded group, recurse into children
            if let TreeNode::Group {
                children, expanded, ..
            } = node
                && *expanded
                && Self::toggle_node_by_index_recursive(target_idx, children, current_idx)
            {
                return true;
            }
        }
        false
    }
}

/// IDE-style "compact folders": collapse chains of single-child groups so a lone
/// deeply-nested tensor (or middle package) shows on one row. Children are
/// compacted first, then a group with exactly one child is folded:
/// - a single sub-group merges its name in (`a` + `b` → `a.b`, keeping `b`'s
///   children);
/// - a single tensor is absorbed into the leaf, whose label becomes the joined
///   path (e.g. `self_attn.k_norm.weight`).
fn compact_nodes(nodes: Vec<TreeNode>) -> Vec<TreeNode> {
    nodes.into_iter().map(compact_node).collect()
}

fn compact_node(node: TreeNode) -> TreeNode {
    let TreeNode::Group {
        name,
        children,
        expanded,
        tensor_count,
        params,
        total_size,
        stored_size,
    } = node
    else {
        return node; // tensors / metadata are leaves
    };
    let mut children = compact_nodes(children);
    if children.len() == 1 {
        match children.pop().unwrap() {
            // Single sub-group: merge names, adopt its (already-compacted)
            // children. The aggregates match (the parent had only this child).
            TreeNode::Group {
                name: child_name,
                children: grandchildren,
                ..
            } => {
                return TreeNode::Group {
                    name: format!("{name}.{child_name}"),
                    children: grandchildren,
                    expanded,
                    tensor_count,
                    params,
                    total_size,
                    stored_size,
                };
            }
            // Single tensor: absorb the group into the leaf's display label.
            TreeNode::Tensor { info, label } => {
                let seg = label.unwrap_or_else(|| last_segment(&info.name).to_string());
                return TreeNode::Tensor {
                    info,
                    label: Some(format!("{name}.{seg}")),
                };
            }
            // Anything else (metadata): leave the group intact.
            other => children.push(other),
        }
    }
    TreeNode::Group {
        name,
        children,
        expanded,
        tensor_count,
        params,
        total_size,
        stored_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 2],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "/tmp/x".to_string(),
            layout: Layout::None,
        }
    }

    /// Find a group by name among `nodes` (non-recursive), returning its children.
    fn group<'a>(nodes: &'a [TreeNode], name: &str) -> &'a [TreeNode] {
        nodes
            .iter()
            .find_map(|n| match n {
                TreeNode::Group {
                    name: g, children, ..
                } if g == name => Some(children.as_slice()),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!(
                    "no group '{name}' in {:?}",
                    nodes.iter().map(|n| n.name()).collect::<Vec<_>>()
                )
            })
    }

    fn leaf_names(nodes: &[TreeNode]) -> Vec<&str> {
        nodes
            .iter()
            .filter_map(|n| match n {
                TreeNode::Tensor { info, .. } => Some(info.name.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn folds_underscore_flattened_names_on_double_underscore() {
        // `.npy` trace names use `__` to separate a tensor from its variant.
        let base = "model_layers_0_block_sparse_moe_experts_down_proj_weight";
        let tree = TreeBuilder::build_tree(&[
            tensor(&format!("{base}__acthost_addr27264")),
            tensor(&format!("{base}__checkpoint")),
            tensor(&format!("{base}__post_transform")),
        ]);
        // One foldable group named after the base, holding the three variants as
        // leaves (still keyed by their full names for lookup).
        let children = group(&tree, base);
        let mut variants = leaf_names(children);
        variants.sort();
        assert_eq!(
            variants,
            vec![
                format!("{base}__acthost_addr27264"),
                format!("{base}__checkpoint"),
                format!("{base}__post_transform"),
            ]
        );
    }

    #[test]
    fn dotted_names_still_fold_on_dots() {
        let tree = TreeBuilder::build_tree(&[
            tensor("model.layers.0.weight"),
            tensor("model.layers.0.bias"),
        ]);
        // The single-child chain model→layers→0 compacts into one group holding
        // both leaves.
        let zero = group(&tree, "model.layers.0");
        let mut names = leaf_names(zero);
        names.sort();
        assert_eq!(names, vec!["model.layers.0.bias", "model.layers.0.weight"]);
    }

    #[test]
    fn compacts_a_single_child_chain_into_one_leaf() {
        let tree = TreeBuilder::build_tree(&[tensor("model.layers.0.self_attn.k_norm.weight")]);
        // A lone deeply-nested tensor becomes a single labelled leaf row.
        assert_eq!(tree.len(), 1);
        match &tree[0] {
            TreeNode::Tensor { info, label } => {
                assert_eq!(info.name, "model.layers.0.self_attn.k_norm.weight");
                assert_eq!(
                    label.as_deref(),
                    Some("model.layers.0.self_attn.k_norm.weight")
                );
            }
            other => panic!("expected one compacted leaf, got group {:?}", other.name()),
        }
    }

    #[test]
    fn compacts_lone_tensors_but_keeps_shared_folders() {
        let tree =
            TreeBuilder::build_tree(&[tensor("enc.a.w"), tensor("enc.b.x"), tensor("enc.b.y")]);
        // `enc` stays (two children). `a` (one tensor) compacts to a leaf
        // labelled `a.w`; `b` (two tensors) stays a group with leaves `x`/`y`.
        let enc = group(&tree, "enc");
        let aw_label = enc.iter().find_map(|n| match n {
            TreeNode::Tensor { info, label } if info.name == "enc.a.w" => Some(label.clone()),
            _ => None,
        });
        assert_eq!(aw_label, Some(Some("a.w".to_string())));
        let b = group(enc, "b");
        let mut names = leaf_names(b);
        names.sort();
        assert_eq!(names, vec!["enc.b.x", "enc.b.y"]);
    }
}
