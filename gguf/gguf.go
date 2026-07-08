// Copyright 2026 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

// Package gguf provides public access to Born's GGUF file format support.
//
// GGUF (GGML Universal Format) is the file format used by llama.cpp for
// storing quantized LLM models. This package exposes the parser, the typed
// metadata accessors, and the tensor converter that dequantizes tensor data
// to float32, so external tools can read GGUF files — and convert models to
// Born's native format — without depending on internal packages.
//
// Format detection is by content, not file extension: ParseFile reads the
// GGUF magic, so extension-less files such as content-addressed cache blobs
// parse fine.
//
// Example:
//
//	file, err := gguf.ParseFile("model.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Architecture: %s\n", file.Architecture())
//
//	converter, err := gguf.NewTensorConverter(file)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer converter.Close()
//
//	for i := range file.TensorInfo {
//	    data, shape, err := converter.Convert(file.TensorInfo[i].Name)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    // data is float32, shape is in Born order (reversed from GGUF).
//	    _ = data
//	    _ = shape
//	}
package gguf

import "github.com/born-ml/born/internal/gguf"

// File represents a parsed GGUF file: header, metadata, tensor index, and
// data-section offsets. It provides typed accessors for the load-bearing
// metadata keys (Architecture, EmbeddingLength, BlockCount, VocabSize, …).
type File = gguf.File

// Header represents the GGUF file header.
type Header = gguf.Header

// TensorInfo contains metadata about a tensor in the file: name,
// dimensions (in GGUF order), GGML type, and offset within the data
// section.
type TensorInfo = gguf.TensorInfo

// GGMLType represents the data type of tensor elements.
type GGMLType = gguf.GGMLType

// TypeTrait contains metadata about a GGML type: block size, bytes per
// block, and whether the type is a quantized format.
type TypeTrait = gguf.TypeTrait

// TensorConverter converts GGUF tensors to float32, dequantizing
// quantized formats (Q4_0, Q4_K, Q8_0, F16, …) transparently. Create one
// per File with NewTensorConverter and close it when done.
type TensorConverter = gguf.TensorConverter

// ConvertedTensor holds a converted tensor: float32 data and the shape in
// Born order (reversed from GGUF).
type ConvertedTensor = gguf.ConvertedTensor

// Magic bytes for the GGUF format.
const (
	MagicGGUFLE = gguf.MagicGGUFLE
	MagicGGUFBE = gguf.MagicGGUFBE
)

// GGUF format versions.
const (
	Version1 = gguf.Version1
	Version2 = gguf.Version2
	Version3 = gguf.Version3 // Current version.
)

// DefaultAlignment is the default alignment of the tensor data section.
const DefaultAlignment = gguf.DefaultAlignment

// GGML tensor types (quantization formats).
//
//nolint:revive // Underscores in names match GGML specification.
const (
	GGMLTypeF32     = gguf.GGMLTypeF32
	GGMLTypeF16     = gguf.GGMLTypeF16
	GGMLTypeQ4_0    = gguf.GGMLTypeQ4_0
	GGMLTypeQ4_1    = gguf.GGMLTypeQ4_1
	GGMLTypeQ5_0    = gguf.GGMLTypeQ5_0
	GGMLTypeQ5_1    = gguf.GGMLTypeQ5_1
	GGMLTypeQ8_0    = gguf.GGMLTypeQ8_0
	GGMLTypeQ8_1    = gguf.GGMLTypeQ8_1
	GGMLTypeQ2_K    = gguf.GGMLTypeQ2_K
	GGMLTypeQ3_K    = gguf.GGMLTypeQ3_K
	GGMLTypeQ4_K    = gguf.GGMLTypeQ4_K
	GGMLTypeQ5_K    = gguf.GGMLTypeQ5_K
	GGMLTypeQ6_K    = gguf.GGMLTypeQ6_K
	GGMLTypeQ8_K    = gguf.GGMLTypeQ8_K
	GGMLTypeIQ2_XXS = gguf.GGMLTypeIQ2_XXS
	GGMLTypeIQ2_XS  = gguf.GGMLTypeIQ2_XS
	GGMLTypeIQ3_XXS = gguf.GGMLTypeIQ3_XXS
	GGMLTypeIQ1_S   = gguf.GGMLTypeIQ1_S
	GGMLTypeIQ4_NL  = gguf.GGMLTypeIQ4_NL
	GGMLTypeIQ3_S   = gguf.GGMLTypeIQ3_S
	GGMLTypeIQ2_S   = gguf.GGMLTypeIQ2_S
	GGMLTypeIQ4_XS  = gguf.GGMLTypeIQ4_XS
	GGMLTypeI8      = gguf.GGMLTypeI8
	GGMLTypeI16     = gguf.GGMLTypeI16
	GGMLTypeI32     = gguf.GGMLTypeI32
	GGMLTypeI64     = gguf.GGMLTypeI64
	GGMLTypeF64     = gguf.GGMLTypeF64
	GGMLTypeBF16    = gguf.GGMLTypeBF16
)

// ParseFile parses the GGUF file at path.
//
// Detection is by content (the GGUF magic), not file extension.
func ParseFile(path string) (*File, error) {
	return gguf.ParseFile(path)
}

// NewTensorConverter creates a TensorConverter for the given file.
//
// The file must have been parsed with ParseFile: the converter reads
// tensor data through the file's recorded path.
func NewTensorConverter(file *File) (*TensorConverter, error) {
	return gguf.NewTensorConverter(file)
}
