<?php
/**
 * Meter tests
 *
 * @package Test
 * @author Andreas Götz <cpuidle@gmx.de>
 */

require_once('DataContext.php');

class DataSensorTest extends DataContext
{
	/**
	 * Create channel
	 */
	static function setupBeforeClass() {
		parent::setupBeforeClass();
		self::$uuid = self::createChannel('Sensor', 'powersensor'/*, self::$resolution*/);
	}

	function testSensor() {
		echo('Not Implemented');
	}
}

?>
