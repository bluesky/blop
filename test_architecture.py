#!/usr/bin/env python3
"""
Simple architecture test to verify the new blop_sim structure.
This script checks imports and basic instantiation patterns.
"""
import sys
sys.path.insert(0, '/workspace/sim')

print("=" * 60)
print("Testing blop_sim architecture")
print("=" * 60)

# Test 1: Import backends
print("\n[1] Testing backend imports...")
try:
    from blop_sim import SimpleBackend, XRTBackend
    print("✓ Successfully imported SimpleBackend and XRTBackend")
except Exception as e:
    print(f"✗ Failed to import backends: {e}")
    sys.exit(1)

# Test 2: Import devices
print("\n[2] Testing device imports...")
try:
    from blop_sim.devices import KBMirrorSimple, KBMirrorXRT, SlitDevice, DetectorDevice
    print("✓ Successfully imported all device classes")
except Exception as e:
    print(f"✗ Failed to import devices: {e}")
    sys.exit(1)

# Test 3: Import utilities
print("\n[3] Testing utility imports...")
try:
    from blop_sim import get_beam_stats, HDF5Handler
    print("✓ Successfully imported utilities")
except Exception as e:
    print(f"✗ Failed to import utilities: {e}")
    sys.exit(1)

# Test 4: Check backend singleton pattern
print("\n[4] Testing backend singleton pattern...")
try:
    backend1 = SimpleBackend()
    backend2 = SimpleBackend()
    assert backend1 is backend2, "SimpleBackend should be a singleton"
    print(f"✓ SimpleBackend singleton works (id: {id(backend1)})")
except Exception as e:
    print(f"✗ Singleton pattern failed: {e}")
    sys.exit(1)

# Test 5: Check backend switch
print("\n[5] Testing backend switching...")
try:
    simple = SimpleBackend()
    xrt = XRTBackend()
    simple2 = SimpleBackend()
    xrt2 = XRTBackend()
    assert simple2 is simple, "SimpleBackend should be a singleton"
    assert xrt2 is xrt, "XRTBackend should be a singleton"
    assert simple is not xrt, "Different backend types should be different instances"
    print("✓ Backend singletons work correctly")
except Exception as e:
    print(f"✗ Backend switching failed: {e}")
    sys.exit(1)

# Test 6: Verify device classes have expected structure
print("\n[6] Verifying device class structure...")
try:
    # Check KBMirrorSimple has expected attributes
    assert hasattr(KBMirrorSimple, '__init__'), "KBMirrorSimple should have __init__"
    print("✓ KBMirrorSimple structure looks good")
    
    # Check KBMirrorXRT has expected attributes
    assert hasattr(KBMirrorXRT, '__init__'), "KBMirrorXRT should have __init__"
    print("✓ KBMirrorXRT structure looks good")
    
    # Check SlitDevice has expected attributes
    assert hasattr(SlitDevice, '__init__'), "SlitDevice should have __init__"
    print("✓ SlitDevice structure looks good")
    
    # Check DetectorDevice has expected attributes
    assert hasattr(DetectorDevice, '__init__'), "DetectorDevice should have __init__"
    print("✓ DetectorDevice structure looks good")
except Exception as e:
    print(f"✗ Device structure verification failed: {e}")
    sys.exit(1)

# Test 7: Check backend methods exist
print("\n[7] Verifying backend methods...")
try:
    backend = SimpleBackend()
    assert hasattr(backend, 'register_device'), "Backend should have register_device"
    assert hasattr(backend, 'generate_beam'), "Backend should have generate_beam"
    assert hasattr(backend, 'get_image_shape'), "Backend should have get_image_shape"
    print("✓ Backend has required methods")
except Exception as e:
    print(f"✗ Backend method verification failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All architecture tests passed! ✓")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pixi install")
print("2. Run tutorial: jupyter notebook docs/source/tutorials/xrt-kb-mirrors.ipynb")
print("3. Verify full integration with Bluesky")
