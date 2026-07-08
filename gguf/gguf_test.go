// Copyright 2026 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package gguf_test

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/born-ml/born/gguf"
)

// GGUF metadata value-type tags (wire format). These are spec-level
// encoding details, not part of the public facade, so the black-box
// test carries its own copy.
const (
	ggufTypeUint32 uint32 = 4
	ggufTypeString uint32 = 8
	ggufTypeArray  uint32 = 9
)

// writeGGUFString writes a GGUF string (uint64 length + UTF-8 bytes).
func writeGGUFString(buf *bytes.Buffer, s string) {
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

// createTestGGUF writes a minimal GGUF v3 file named fileName into a
// temporary directory and returns its path. The file carries three
// metadata entries (architecture, embedding length, and a string array —
// real models always carry array metadata such as tokenizer.ggml.tokens)
// and one F32 tensor "blk.0.attn_q.weight" of GGUF dimensions [3, 2]
// holding the values 1..6.
func createTestGGUF(t *testing.T, fileName string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), fileName)
	buf := new(bytes.Buffer)
	order := binary.LittleEndian

	// Header.
	_ = binary.Write(buf, order, gguf.MagicGGUFLE)
	_ = binary.Write(buf, order, gguf.Version3)
	_ = binary.Write(buf, order, uint64(1)) // Tensor count.
	_ = binary.Write(buf, order, uint64(3)) // Metadata count.

	// general.architecture = "llama".
	writeGGUFString(buf, "general.architecture")
	_ = binary.Write(buf, order, ggufTypeString)
	writeGGUFString(buf, "llama")

	// llama.embedding_length = 64.
	writeGGUFString(buf, "llama.embedding_length")
	_ = binary.Write(buf, order, ggufTypeUint32)
	_ = binary.Write(buf, order, uint32(64))

	// tokenizer.ggml.tokens = ["<s>", "</s>"].
	writeGGUFString(buf, "tokenizer.ggml.tokens")
	_ = binary.Write(buf, order, ggufTypeArray)
	_ = binary.Write(buf, order, ggufTypeString)
	_ = binary.Write(buf, order, uint64(2))
	writeGGUFString(buf, "<s>")
	writeGGUFString(buf, "</s>")

	// Tensor info: "blk.0.attn_q.weight", dims [3, 2] (GGUF order), F32.
	writeGGUFString(buf, "blk.0.attn_q.weight")
	_ = binary.Write(buf, order, uint32(2))
	_ = binary.Write(buf, order, uint64(3))
	_ = binary.Write(buf, order, uint64(2))
	_ = binary.Write(buf, order, gguf.GGMLTypeF32)
	_ = binary.Write(buf, order, uint64(0))

	// Padding to the tensor data section.
	for buf.Len()%gguf.DefaultAlignment != 0 {
		buf.WriteByte(0)
	}

	// Tensor data: 6 float32 values 1..6.
	for i := 1; i <= 6; i++ {
		_ = binary.Write(buf, order, float32(i))
	}

	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write test gguf: %v", err)
	}
	return path
}

func TestParseFile(t *testing.T) {
	path := createTestGGUF(t, "model.gguf")

	file, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}

	if file.Header.Version != gguf.Version3 {
		t.Errorf("version: got %d, want %d", file.Header.Version, gguf.Version3)
	}
	if got := file.Architecture(); got != "llama" {
		t.Errorf("architecture: got %q, want %q", got, "llama")
	}
	if got := file.EmbeddingLength(); got != 64 {
		t.Errorf("embedding length: got %d, want 64", got)
	}
	if got := len(file.TensorInfo); got != 1 {
		t.Fatalf("tensor count: got %d, want 1", got)
	}

	info := file.GetTensor("blk.0.attn_q.weight")
	if info == nil {
		t.Fatal("GetTensor: tensor not found")
	}
	if info.Type != gguf.GGMLTypeF32 {
		t.Errorf("tensor type: got %v, want %v", info.Type, gguf.GGMLTypeF32)
	}
	if got := info.NumElements(); got != 6 {
		t.Errorf("elements: got %d, want 6", got)
	}
}

// TestParseFile_ArrayMetadata verifies that array-typed metadata parses.
// Every real model carries arrays (e.g. tokenizer.ggml.tokens), so this
// is load-bearing for opening real GGUF files.
func TestParseFile_ArrayMetadata(t *testing.T) {
	path := createTestGGUF(t, "model.gguf")

	file, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}

	tokens, ok := file.Metadata["tokenizer.ggml.tokens"].([]string)
	if !ok {
		t.Fatalf("tokens metadata: got %T, want []string",
			file.Metadata["tokenizer.ggml.tokens"])
	}
	if len(tokens) != 2 || tokens[0] != "<s>" || tokens[1] != "</s>" {
		t.Errorf("tokens: got %v, want [<s> </s>]", tokens)
	}
}

// TestParseFile_NoExtension verifies content-based format detection:
// a GGUF file without the .gguf extension — such as a content-addressed
// cache blob — parses fine.
func TestParseFile_NoExtension(t *testing.T) {
	path := createTestGGUF(t, "sha256-0011223344556677")

	file, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("ParseFile without extension: %v", err)
	}
	if got := file.Architecture(); got != "llama" {
		t.Errorf("architecture: got %q, want %q", got, "llama")
	}
}

func TestParseFile_InvalidMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "not-gguf.bin")
	if err := os.WriteFile(path, []byte("this is not a GGUF file"), 0o600); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	if _, err := gguf.ParseFile(path); err == nil {
		t.Error("ParseFile: expected error for non-GGUF content, got nil")
	}
}

func TestTensorConverter(t *testing.T) {
	path := createTestGGUF(t, "model.gguf")

	file, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}

	converter, err := gguf.NewTensorConverter(file)
	if err != nil {
		t.Fatalf("NewTensorConverter: %v", err)
	}
	defer func() { _ = converter.Close() }()

	data, shape, err := converter.Convert("blk.0.attn_q.weight")
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	// Shape comes back in Born order: GGUF [3, 2] → Born [2, 3].
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape: got %v, want [2 3]", shape)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	if len(data) != len(want) {
		t.Fatalf("data length: got %d, want %d", len(data), len(want))
	}
	for i := range want {
		if data[i] != want[i] {
			t.Errorf("data[%d]: got %v, want %v", i, data[i], want[i])
		}
	}
}

func TestGGMLType(t *testing.T) {
	if got := gguf.GGMLTypeQ4_K.String(); got != "Q4_K" {
		t.Errorf("String: got %q, want %q", got, "Q4_K")
	}
	if !gguf.GGMLTypeQ4_K.IsQuantized() {
		t.Error("IsQuantized: Q4_K should be quantized")
	}
	if gguf.GGMLTypeF32.IsQuantized() {
		t.Error("IsQuantized: F32 should not be quantized")
	}
	if got := gguf.GGMLTypeF32.RowSize(6); got != 24 {
		t.Errorf("RowSize: got %d, want 24", got)
	}
}
