<?php

namespace Volkszaehler\Model\Proxy;

/**
 * THIS CLASS WAS GENERATED BY THE DOCTRINE ORM. DO NOT EDIT THIS FILE.
 */
class VolkszaehlerModelAggregatorProxy extends \Volkszaehler\Model\Aggregator implements \Doctrine\ORM\Proxy\Proxy
{
    private $_entityPersister;
    private $_identifier;
    public $__isInitialized__ = false;
    public function __construct($entityPersister, $identifier)
    {
        $this->_entityPersister = $entityPersister;
        $this->_identifier = $identifier;
    }
    private function _load()
    {
        if (!$this->__isInitialized__ && $this->_entityPersister) {
            $this->__isInitialized__ = true;
            if ($this->_entityPersister->load($this->_identifier, $this) === null) {
                throw new \Doctrine\ORM\EntityNotFoundException();
            }
            unset($this->_entityPersister, $this->_identifier);
        }
    }

    
    public function addChild(\Volkszaehler\Model\Entity $child)
    {
        $this->_load();
        return parent::addChild($child);
    }

    public function removeChild(\Volkszaehler\Model\Entity $child)
    {
        $this->_load();
        return parent::removeChild($child);
    }

    public function getChildren()
    {
        $this->_load();
        return parent::getChildren();
    }

    public function checkProperties()
    {
        $this->_load();
        return parent::checkProperties();
    }

    public function getProperty($key)
    {
        $this->_load();
        return parent::getProperty($key);
    }

    public function getProperties($prefix = NULL)
    {
        $this->_load();
        return parent::getProperties($prefix);
    }

    public function setProperty($key, $value)
    {
        $this->_load();
        return parent::setProperty($key, $value);
    }

    public function unsetProperty($key, \Doctrine\ORM\EntityManager $em)
    {
        $this->_load();
        return parent::unsetProperty($key, $em);
    }

    public function getId()
    {
        $this->_load();
        return parent::getId();
    }

    public function getUuid()
    {
        $this->_load();
        return parent::getUuid();
    }

    public function getType()
    {
        $this->_load();
        return parent::getType();
    }

    public function getDefinition()
    {
        $this->_load();
        return parent::getDefinition();
    }

    public function getInterpreter(\Doctrine\ORM\EntityManager $em, $from, $to)
    {
        $this->_load();
        return parent::getInterpreter($em, $from, $to);
    }


    public function __sleep()
    {
        return array('__isInitialized__', 'id', 'uuid', 'type', 'tokens', 'properties', 'parents', 'children');
    }
}