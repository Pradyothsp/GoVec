package response

import "github.com/gin-gonic/gin"

// Response is the standard JSON envelope for all API responses.
type Response struct {
	Success bool   `json:"success" example:"true"`
	Data    any    `json:"data"`
	Error   string `json:"error,omitempty" example:""`
}

// ErrorResponse is the standard JSON envelope for failed API responses.
// Currently only used for Swagger response
type ErrorResponse struct {
	Success bool   `json:"success" example:"false"`
	Error   string `json:"error" example:"error message"`
}

// OK writes a success response with the given HTTP status code and data payload.
func OK(c *gin.Context, status int, data any) {
	c.JSON(status, Response{Success: true, Data: data})
}

// Fail writes an error response with the given HTTP status code and error message.
func Fail(c *gin.Context, status int, message string) {
	c.JSON(status, Response{Success: false, Error: message})
}
