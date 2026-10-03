package requestid

import (
	"crypto/rand"
	"fmt"
)

// GinKey is the key used to store the correlation ID in gin.Context.
const GinKey = "correlation_id"

// Generate returns a random RFC 4122 v4 UUID string.
// It uses crypto/rand, so IDs are unpredictable and safe to expose to clients.
func Generate() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("requestId: crypto/rand.Read failed: " + err.Error())
	}

	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant bits

	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
