_MAJOR = "2024"
_MINOR = "1"
# On main and in a nightly release the patch should be one ahead of the last
# released build.
_PATCH = "2"
# This is mainly for nightly builds which have the suffix ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
_SUFFIX = ""

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
__version__ = VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH, _SUFFIX)

print(VERSION, VERSION_SHORT)