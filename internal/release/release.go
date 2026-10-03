// Package release holds facts about the running binary that are fixed at build time.
package release

// Version is the release this binary was built from. The release image stamps it with the git
// tag and `task build` with `git describe` (-ldflags "-X <this package>.Version=v0.1.0"); a
// plain `go build` reports "dev". Both transports return it from Info, and the server logs it
// at startup.
var Version = "dev"
